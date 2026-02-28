package com.garbage.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import java.util.LinkedList

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()
    private var bounds = Rect()

    init {
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = Color.GREEN
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (result in results) {
            val left = result.x1 * width
            val top = result.y1 * height
            val right = result.x2 * width
            val bottom = result.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)
            
            val drawableText = result.clsName + " " + String.format("%.2f", result.cnf)
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            
            canvas.drawRect(
                left,
                top,
                left + textWidth + 8,
                top + textHeight + 8,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + textHeight, textPaint)
        }
    }
}