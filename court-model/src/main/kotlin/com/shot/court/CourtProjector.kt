package com.shot.court

import com.shot.core.ItfCourtSpec
import com.shot.core.model.Keypoint

/**
 * Projects all 16 court keypoints to image coordinates using the inverse homography.
 */
class CourtProjector(
    private val homographyCalculator: HomographyCalculator = HomographyCalculator()
) {

    /**
     * Project all 16 ITF court keypoints to image pixel coordinates.
     *
     * @param homography 3x3 homography matrix (image → court), row-major FloatArray
     * @return List of 16 Keypoint objects with image coordinates, or empty if H is invalid
     */
    fun projectAllKeypoints(homography: FloatArray): List<Keypoint> {
        val hInv = homographyCalculator.invertHomography(homography) ?: return emptyList()

        return ItfCourtSpec.KEYPOINTS.map { (id, courtPoint) ->
            val imageCoords = homographyCalculator.projectCourtToImage(
                courtPoint.x, courtPoint.y, hInv
            )
            Keypoint(
                id = id,
                x = imageCoords[0],
                y = imageCoords[1],
                confidence = if (imageCoords[0].isNaN()) 0f else 1f
            )
        }.filter { !it.x.isNaN() && !it.y.isNaN() }
    }

    /**
     * Compute reprojection error for detected keypoints.
     * Projects detected keypoints through H → court → H_inv → image and measures the error.
     *
     * @param detectedKeypoints Original detected keypoints (image coordinates)
     * @param homography 3x3 homography matrix (image → court)
     * @return Mean reprojection error in pixels, or Float.MAX_VALUE if invalid
     */
    fun computeReprojectionError(
        detectedKeypoints: List<Keypoint>,
        homography: FloatArray
    ): Float {
        val hInv = homographyCalculator.invertHomography(homography) ?: return Float.MAX_VALUE

        var totalError = 0f
        var count = 0

        for (kp in detectedKeypoints) {
            val courtPoint = ItfCourtSpec.KEYPOINTS[kp.id] ?: continue

            // Project court coord back to image
            val reprojected = homographyCalculator.projectCourtToImage(
                courtPoint.x, courtPoint.y, hInv
            )

            if (reprojected[0].isNaN()) continue

            // Euclidean distance between detected and reprojected
            val dx = kp.x - reprojected[0]
            val dy = kp.y - reprojected[1]
            totalError += Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            count++
        }

        return if (count > 0) totalError / count else Float.MAX_VALUE
    }
}
