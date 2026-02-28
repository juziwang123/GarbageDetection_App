package com.example.garbagedetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val listener: DetectorListener
) {
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)) // YOLO input size
        .add(NormalizeOp(0f, 255f)) // Normalize 0-255 -> 0-1
        .add(CastOp(DataType.FLOAT32))
        .build()

    fun setup() {
        try {
            val model = FileUtil.loadMappedFile(context, modelPath)
            val options = Interpreter.Options()
            options.numThreads = 4
            interpreter = Interpreter(model, options)

            val inputShape = interpreter?.getInputTensor(0)?.shape() // [1, 224, 224, 3] or [1, 3, 224, 224]
            val outputShape = interpreter?.getOutputTensor(0)?.shape() // [1, 16, 1029] (usually [1, 4+nc, anchors])

            tensorWidth = inputShape?.get(2) ?: 224
            tensorHeight = inputShape?.get(1) ?: 224 
            
            // Check Input format: usually BHWC for TFLite, but sometimes BCHW depending on export.
            // YOLO export usually stays BCHW if exported via ultralytics, OR BHWC if using specific tflite flags.
            // Let's assume standard float32 export via ultralytics keeps input layout dynamic or BHWC. 
            // In image processing, we'll verify.
            
            println("Model Input Shape: ${inputShape?.contentToString()}")
            println("Model Output Shape: ${outputShape?.contentToString()}")

            labels = FileUtil.loadLabels(context, labelPath)
            
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun detect(image: Bitmap) {
        interpreter?.let { model ->
            val startTime = SystemClock.uptimeMillis()
            
            // Preprocess
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(image)
            val processedImage = imageProcessor.process(tensorImage)
            val imageBuffer = processedImage.buffer

            // Output buffer: [1, 16, 1029]
            // We need to clarify if it's [1, anchors, 4+nc] or [1, 4+nc, anchors].
            // Ultralytics usually exports as [1, 4+nc, anchors].
            // We will transpose it if needed or read accordingly.
            val outputTensor = TensorBuffer.createFixedSize(intArrayOf(1, 16, 1029), DataType.FLOAT32)
            
            // Run inference
            model.run(imageBuffer, outputTensor.buffer)

            val output = outputTensor.floatArray
            val boundingBoxes = parseYoloOutput(output)
            
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            listener.onDetect(boundingBoxes, inferenceTime)
        }
    }

    // Parsing logic for [1, 4+nc, anchors] -> [1, 16, 1029]
    private fun parseYoloOutput(output: FloatArray): List<BoundingBox> {
        val boxes = mutableListOf<BoundingBox>()
        val numClasses = 12
        val numAnchors = 1029
        val numElements = 16 // 4 + 12

        // Output index mapping:
        // Rows = 16 (cx, cy, w, h, c1, c2, ..., c12)
        // Cols = 1029 (anchors)

        for (i in 0 until numAnchors) {
            // Read column i
            val cx = output[0 * numAnchors + i]
            val cy = output[1 * numAnchors + i]
            val w  = output[2 * numAnchors + i]
            val h  = output[3 * numAnchors + i]

            // Find best class
            var maxConf = 0f
            var maxClass = -1
            for (c in 0 until numClasses) {
                val conf = output[(4 + c) * numAnchors + i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClass = c
                }
            }

            if (maxConf > 0.4f) { // Confidence Threshold
                boxes.add(BoundingBox(
                    x1 = cx - w / 2,
                    y1 = cy - h / 2,
                    x2 = cx + w / 2,
                    y2 = cy + h / 2,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = maxConf,
                    cls = maxClass,
                    clsName = labels.getOrElse(maxClass) { "Unknown" }
                ))
            }
        }
        return applyNMS(boxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sorted = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selected = mutableListOf<BoundingBox>()
        
        while (sorted.isNotEmpty()) {
            val first = sorted.removeAt(0)
            selected.add(first)
            
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(first, next) >= 0.45f) { // IoU Threshold
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        
        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val unionArea = (a.w * a.h) + (b.w * b.h) - intersectionArea
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    data class BoundingBox(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val cnf: Float, val cls: Int, val clsName: String
    )

    interface DetectorListener {
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
        fun onEmptyDetect()
    }
}
