package com.garbage.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val listener: DetectorListener
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var isChannelsFirst = true
    private var imageProcessor: ImageProcessor? = null

    fun setup() {
        try {
            val model = FileUtil.loadMappedFile(context, modelPath)
            val options = Interpreter.Options()
            options.setNumThreads(4)
            interpreter = Interpreter(model, options)

            val inputShape = interpreter?.getInputTensor(0)?.shape() // [1, 224, 224, 3] or [1, 3, 224, 224]
            val outputShape = interpreter?.getOutputTensor(0)?.shape() // [1, 16, 1029] or similar

            // If input is BCHW [1, 3, 224, 224], then width/height are at index 2,3
            if (inputShape != null) {
                if (inputShape[1] == 3) {
                     tensorWidth = inputShape[2]
                     tensorHeight = inputShape[3]
                } else {
                     tensorWidth = inputShape[1]
                     tensorHeight = inputShape[2]
                }
            } else {
                tensorWidth = Constants.INPUT_SIZE
                tensorHeight = Constants.INPUT_SIZE
            }

            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(tensorHeight, tensorWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // Normalize to [0, 1]
                .add(CastOp(org.tensorflow.lite.DataType.FLOAT32))
                .build()

            println("Model Input Shape: ${inputShape?.contentToString()}")
            println("Model Output Shape: ${outputShape?.contentToString()}")

            if (outputShape != null) {
                // Determine if [1, 16, N] or [1, N, 16]
                // 16 is small (channels), N is large (anchors, e.g. 1029 or 8400)
                val dim1 = outputShape[1]
                val dim2 = outputShape[2]
                
                if (dim1 < dim2) {
                    // [1, 16, 1029] -> Channels First (Standard YOLO export)
                    numChannel = dim1
                    numElements = dim2
                    isChannelsFirst = true
                } else {
                     // [1, 1029, 16] -> Channels Last (TFLite sometimes does this)
                     numChannel = dim2
                     numElements = dim1
                     isChannelsFirst = false
                }
            }

            // Load labels
            val reader = BufferedReader(InputStreamReader(context.assets.open(labelPath)))
            var line: String? = reader.readLine()
            while (line != null) {
                if (line.isNotEmpty()) labels.add(line)
                line = reader.readLine()
            }
            reader.close()

        } catch (e: Exception) {
            e.printStackTrace()
            listener.onError("Error init model: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap) {
        val processor = imageProcessor ?: return
        interpreter?.let { tflite ->
            val tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = processor.process(tensorImage)
            val imageBuffer = processedImage.buffer

            // Output buffer
            val shape = if (isChannelsFirst) {
                intArrayOf(1, numChannel, numElements)
            } else {
                intArrayOf(1, numElements, numChannel)
            }
            
            val outputTensorBuffer = TensorBuffer.createFixedSize(
                shape,
                org.tensorflow.lite.DataType.FLOAT32
            )
            
            tflite.run(imageBuffer, outputTensorBuffer.buffer)
            
            val outputFloats = outputTensorBuffer.floatArray
            val bestBoxes = bestBox(outputFloats)
            
            // Should release tensor buffers? ImageProxy is critical, tensor buffers are managed by TFLite.
            
            listener.onDetect(bestBoxes, System.currentTimeMillis()) 
        }
    }

    private fun bestBox(array: FloatArray): List<BoundingBox> {
        val boxes = mutableListOf<BoundingBox>()
        val numClasses = numChannel - 4
        
        for (i in 0 until numElements) {
            val cx: Float
            val cy: Float
            val w: Float
            val h: Float
            
            if (isChannelsFirst) {
                // [1, 16, 1029] -> layout is [cx_all, cy_all, w_all, h_all, class0_all...]
                // array[channel * numElements + element]
                cx = array[0 * numElements + i]
                cy = array[1 * numElements + i]
                w  = array[2 * numElements + i]
                h  = array[3 * numElements + i]
            } else {
                // [1, 1029, 16] -> layout is [anchor0, anchor1...] where anchor0=[cx,cy,w,h,c0...]
                // array[element * numChannel + channel]
                val offset = i * numChannel
                cx = array[offset + 0]
                cy = array[offset + 1]
                w  = array[offset + 2]
                h  = array[offset + 3]
            }
            
            var maxConf = 0f
            var maxClass = -1
            
            for (c in 0 until numClasses) {
                val conf = if (isChannelsFirst) {
                    array[(4 + c) * numElements + i]
                } else {
                    array[i * numChannel + 4 + c]
                }
                
                if (conf > maxConf) {
                    maxConf = conf
                    maxClass = c
                }
            }

            if (maxConf > 0.45f) { // Increased Confidence threshold for stability
                val clsName = if (maxClass in labels.indices) labels[maxClass] else "Unknown($maxClass)"
                
                // Convert to Bounds
                // If model output is normalized (0..1), simple arithmetic.
                // If model output is pixels (0..640), normalize by model input size (e.g. 640).
                // Let's assume normalized first as per web app.
                
                val x1 = (cx - w / 2F)
                val y1 = (cy - h / 2F)
                val x2 = (cx + w / 2F)
                val y2 = (cy + h / 2F)
                
                // If coordinates are clearly > 1.0 (e.g. pixel coords), normalize them
                // But if they are 0..1, keep them.
                // Hard to tell without seeing values. 
                // But web app worked with normalized logic.
                
                boxes.add(BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxConf, maxClass, clsName))
            }
        }

        if (boxes.isEmpty()) return emptyList()

        return nms(boxes)
    }

    private fun nms(boxes: MutableList<BoundingBox>): List<BoundingBox> {
        val finalBoxes = mutableListOf<BoundingBox>()
        boxes.sortByDescending { it.cnf }

        while (boxes.isNotEmpty()) {
            val best = boxes.removeAt(0)
            finalBoxes.add(best)
            
            val iterator = boxes.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(best, next) > 0.45f) {
                    iterator.remove()
                }
            }
        }
        return finalBoxes
    }

    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)

        if (x1 >= x2 || y1 >= y2) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val areaA = a.w * a.h
        val areaB = b.w * b.h
        return intersection / (areaA + areaB - intersection)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }
}