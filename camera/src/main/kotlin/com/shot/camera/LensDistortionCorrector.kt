package com.shot.camera

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager as Camera2Manager
import android.os.Build
import android.util.Log

/**
 * Lens distortion corrector using Camera2 intrinsics.
 *
 * Queries device camera calibration data (focal length, distortion coefficients)
 * and precomputes a remap LUT to undistort frames before court keypoint detection.
 *
 * Court detection runs once at startup, so speed is not critical.
 * After court is locked, call release() to free LUT memory (~7MB).
 */
class LensDistortionCorrector(context: Context, private val width: Int, private val height: Int) {

    companion object {
        private const val TAG = "LensDistortion"

        // Conservative fallback for typical phone barrel distortion
        private const val FALLBACK_K1 = -0.02f
        private const val FALLBACK_K2 = 0.0f
        private const val FALLBACK_P1 = 0.0f
        private const val FALLBACK_P2 = 0.0f
        private const val FALLBACK_K3 = 0.0f

        // Skip undistortion if max pixel displacement < this threshold
        private const val SIGNIFICANCE_THRESHOLD = 1.5f
    }

    // Camera intrinsics
    private val fx: Float
    private val fy: Float
    private val cx: Float
    private val cy: Float
    private val k1: Float
    private val k2: Float
    private val p1: Float
    private val p2: Float
    private val k3: Float

    val isDeviceCalibrated: Boolean

    // Precomputed remap LUT
    private var mapX: FloatArray? = null
    private var mapY: FloatArray? = null

    private val significantDistortion: Boolean

    init {
        val calibration = queryCalibration(context)
        fx = calibration.fx
        fy = calibration.fy
        cx = calibration.cx
        cy = calibration.cy
        k1 = calibration.k1
        k2 = calibration.k2
        p1 = calibration.p1
        p2 = calibration.p2
        k3 = calibration.k3
        isDeviceCalibrated = calibration.fromDevice

        // Build remap LUT
        buildRemapTable()
        significantDistortion = computeSignificance()

        Log.d(TAG, "Initialized: fx=$fx fy=$fy cx=$cx cy=$cy k1=$k1 k2=$k2 " +
                "fromDevice=$isDeviceCalibrated significant=$significantDistortion")
    }

    private data class Calibration(
        val fx: Float, val fy: Float,
        val cx: Float, val cy: Float,
        val k1: Float, val k2: Float,
        val p1: Float, val p2: Float, val k3: Float,
        val fromDevice: Boolean
    )

    private fun queryCalibration(context: Context): Calibration {
        try {
            val camManager = context.getSystemService(Context.CAMERA_SERVICE) as Camera2Manager
            val backCameraId = camManager.cameraIdList.firstOrNull { id ->
                val chars = camManager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: return fallbackCalibration()

            val chars = camManager.getCameraCharacteristics(backCameraId)

            // Query intrinsics (API 23+)
            val intrinsics = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)

            // Query distortion (API 28+)
            val distortion = if (Build.VERSION.SDK_INT >= 28) {
                chars.get(CameraCharacteristics.LENS_DISTORTION)
            } else {
                null
            }

            if (intrinsics != null && intrinsics.size >= 5) {
                val qfx = intrinsics[0]
                val qfy = intrinsics[1]
                val qcx = intrinsics[2]
                val qcy = intrinsics[3]

                // Validate: fx/fy must be positive and reasonable
                if (qfx <= 0f || qfy <= 0f || qfx > 10000f) {
                    Log.w(TAG, "Invalid intrinsics: fx=$qfx fy=$qfy, using fallback")
                    return fallbackCalibration()
                }

                // Scale intrinsics from sensor resolution to our target resolution
                // Camera2 intrinsics are in sensor pixel coordinates
                val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
                val scaleX = if (sensorSize != null) width.toFloat() / sensorSize.width else 1f
                val scaleY = if (sensorSize != null) height.toFloat() / sensorSize.height else 1f

                val scaledFx = qfx * scaleX
                val scaledFy = qfy * scaleY
                val scaledCx = qcx * scaleX
                val scaledCy = qcy * scaleY

                val dk1: Float; val dk2: Float; val dp1: Float; val dp2: Float; val dk3: Float
                if (distortion != null && distortion.size >= 5) {
                    dk1 = distortion[0]
                    dk2 = distortion[1]
                    dp1 = distortion[2]
                    dp2 = distortion[3]
                    dk3 = distortion[4]
                } else {
                    dk1 = FALLBACK_K1; dk2 = FALLBACK_K2
                    dp1 = FALLBACK_P1; dp2 = FALLBACK_P2; dk3 = FALLBACK_K3
                }

                return Calibration(scaledFx, scaledFy, scaledCx, scaledCy,
                    dk1, dk2, dp1, dp2, dk3, fromDevice = true)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Camera2 query failed: ${e.message}")
        }

        return fallbackCalibration()
    }

    private fun fallbackCalibration(): Calibration {
        // Typical phone camera: focal length ≈ larger image dimension
        val f = maxOf(width, height).toFloat()
        return Calibration(
            fx = f, fy = f,
            cx = width / 2f, cy = height / 2f,
            k1 = FALLBACK_K1, k2 = FALLBACK_K2,
            p1 = FALLBACK_P1, p2 = FALLBACK_P2, k3 = FALLBACK_K3,
            fromDevice = false
        )
    }

    /**
     * Precompute remap table: for each undistorted output pixel,
     * find the corresponding distorted source pixel.
     */
    private fun buildRemapTable() {
        val numPixels = width * height
        val mx = FloatArray(numPixels)
        val my = FloatArray(numPixels)

        for (v in 0 until height) {
            for (u in 0 until width) {
                val idx = v * width + u

                // Normalize to camera coordinates
                val x = (u - cx) / fx
                val y = (v - cy) / fy

                // Radial distance
                val r2 = x * x + y * y
                val r4 = r2 * r2
                val r6 = r4 * r2

                // Radial distortion
                val radial = 1f + k1 * r2 + k2 * r4 + k3 * r6

                // Tangential distortion
                val dx = 2f * p1 * x * y + p2 * (r2 + 2f * x * x)
                val dy = p1 * (r2 + 2f * y * y) + 2f * p2 * x * y

                // Distorted normalized coordinates
                val xd = x * radial + dx
                val yd = y * radial + dy

                // Back to pixel coordinates
                mx[idx] = xd * fx + cx
                my[idx] = yd * fy + cy
            }
        }

        mapX = mx
        mapY = my
    }

    /**
     * Check if distortion is significant enough to warrant correction.
     */
    private fun computeSignificance(): Boolean {
        val mx = mapX ?: return false
        val my = mapY ?: return false
        var maxDisp = 0f

        for (v in 0 until height) {
            for (u in 0 until width) {
                val idx = v * width + u
                val dispX = mx[idx] - u
                val dispY = my[idx] - v
                val disp = Math.sqrt((dispX * dispX + dispY * dispY).toDouble()).toFloat()
                if (disp > maxDisp) maxDisp = disp
            }
        }

        Log.d(TAG, "Max pixel displacement: ${"%.2f".format(maxDisp)}px")
        return maxDisp > SIGNIFICANCE_THRESHOLD
    }

    /**
     * Returns true if lens distortion is significant enough to warrant correction.
     */
    fun hasSignificantDistortion(): Boolean = significantDistortion

    /**
     * Undistort a bitmap using the precomputed remap table.
     * Returns a new bitmap; caller should recycle the input.
     */
    fun undistort(bitmap: Bitmap): Bitmap {
        val mx = mapX ?: return bitmap
        val my = mapY ?: return bitmap

        val srcPixels = IntArray(width * height)
        bitmap.getPixels(srcPixels, 0, width, 0, 0, width, height)

        val dstPixels = IntArray(width * height)

        for (i in dstPixels.indices) {
            val srcX = mx[i]
            val srcY = my[i]

            // Bilinear interpolation
            val x0 = srcX.toInt().coerceIn(0, width - 2)
            val y0 = srcY.toInt().coerceIn(0, height - 2)
            val x1 = x0 + 1
            val y1 = y0 + 1

            val fx = srcX - x0
            val fy = srcY - y0

            val p00 = srcPixels[y0 * width + x0]
            val p10 = srcPixels[y0 * width + x1]
            val p01 = srcPixels[y1 * width + x0]
            val p11 = srcPixels[y1 * width + x1]

            // Interpolate each channel
            val a = interpolateChannel(p00, p10, p01, p11, fx, fy, 24)  // A
            val r = interpolateChannel(p00, p10, p01, p11, fx, fy, 16)  // R
            val g = interpolateChannel(p00, p10, p01, p11, fx, fy, 8)   // G
            val b = interpolateChannel(p00, p10, p01, p11, fx, fy, 0)   // B

            dstPixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(dstPixels, 0, width, 0, 0, width, height)
        return result
    }

    private fun interpolateChannel(p00: Int, p10: Int, p01: Int, p11: Int,
                                   fx: Float, fy: Float, shift: Int): Int {
        val c00 = (p00 shr shift) and 0xFF
        val c10 = (p10 shr shift) and 0xFF
        val c01 = (p01 shr shift) and 0xFF
        val c11 = (p11 shr shift) and 0xFF

        val top = c00 + (c10 - c00) * fx
        val bot = c01 + (c11 - c01) * fx
        return (top + (bot - top) * fy).toInt().coerceIn(0, 255)
    }

    /**
     * Release LUT memory after court detection is locked.
     */
    fun release() {
        mapX = null
        mapY = null
        Log.d(TAG, "LUT released (~7MB freed)")
    }
}
