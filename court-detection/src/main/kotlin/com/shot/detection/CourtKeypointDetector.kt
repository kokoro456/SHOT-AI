package com.shot.detection

import android.content.Context
import android.graphics.Bitmap
import com.shot.core.model.Keypoint
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Detects tennis court keypoints using a TFLite model.
 *
 * Loads the court_keypoint.tflite model from assets and runs inference
 * on camera frames to detect 8 keypoints (points 9-16) on the near court.
 */
class CourtKeypointDetector(context: Context) {

    companion object {
        private const val MODEL_FILE = "court_keypoint.tflite"
        private const val INPUT_SIZE = 256
        private const val NUM_KEYPOINTS = 8
        private const val OUTPUT_SIZE = NUM_KEYPOINTS * 3 // x, y, confidence
        private const val NUM_THREADS = 4

        // ImageNet normalization constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Keypoint IDs in model output order
        private val KEYPOINT_IDS = intArrayOf(9, 10, 11, 12, 13, 14, 15, 16)
    }

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer

    init {
        val model = FileUtil.loadMappedFile(context, MODEL_FILE)
        val options = Interpreter.Options().apply {
            numThreads = NUM_THREADS
        }
        interpreter = Interpreter(model, options)

        // Allocate buffers (NHWC format: 1 × H × W × 3)
        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        outputBuffer = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
    }

    /**
     * Detect keypoints from a camera frame bitmap.
     *
     * @param bitmap Camera frame (any size, will be resized to 256x256)
     * @param imageWidth Original image width (for coordinate scaling)
     * @param imageHeight Original image height (for coordinate scaling)
     * @return List of 8 Keypoint objects with pixel coordinates and confidence
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): List<Keypoint> {
        val startTime = System.nanoTime()

        // Preprocess
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        preprocessBitmap(resized, inputBuffer)

        // Inference
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        // Postprocess
        val keypoints = parseOutput(outputBuffer, imageWidth, imageHeight)

        val elapsed = (System.nanoTime() - startTime) / 1_000_000
        return keypoints
    }

    /**
     * Get the inference time of the last detection (for debug display).
     */
    var lastInferenceTimeMs: Long = 0
        private set

    /**
     * Convert a bitmap to the model input format.
     * NHWC format (TFLite convention), ImageNet normalized, float32.
     */
    private fun preprocessBitmap(bitmap: Bitmap, buffer: ByteBuffer) {
        buffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Convert to NHWC float32 with ImageNet normalization
        // Pixel order: for each pixel, write R, G, B channels sequentially
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16 and 0xFF) / 255f
            val g = (pixel shr 8 and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            buffer.putFloat((r - MEAN[0]) / STD[0])
            buffer.putFloat((g - MEAN[1]) / STD[1])
            buffer.putFloat((b - MEAN[2]) / STD[2])
        }
    }

    /**
     * Parse model output buffer into Keypoint objects.
     * Output format: [x0, y0, conf0, x1, y1, conf1, ...] (24 float values)
     * x, y are normalized [0, 1], confidence is [0, 1] (sigmoid activated in model)
     */
    private fun parseOutput(buffer: ByteBuffer, imageWidth: Int, imageHeight: Int): List<Keypoint> {
        buffer.rewind()
        val values = FloatArray(OUTPUT_SIZE)
        buffer.asFloatBuffer().get(values)

        return (0 until NUM_KEYPOINTS).map { i ->
            val x = values[i * 3] * imageWidth       // scale to image width
            val y = values[i * 3 + 1] * imageHeight  // scale to image height
            val confidence = values[i * 3 + 2]

            Keypoint(
                id = KEYPOINT_IDS[i],
                x = x,
                y = y,
                confidence = confidence
            )
        }
    }

    fun close() {
        interpreter.close()
    }
}
