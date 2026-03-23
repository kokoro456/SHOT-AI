package com.shot.court

import com.shot.core.model.Keypoint
import kotlin.math.abs

/**
 * Validates homography matrices and filters geometrically invalid projections.
 */
class HomographyValidator(
    private val projector: CourtProjector = CourtProjector()
) {

    companion object {
        const val MAX_REPROJECTION_ERROR = 35f    // 50→35: 더 엄격한 검증
    }

    fun isValid(homography: FloatArray, detectedKeypoints: List<Keypoint>): Boolean {
        // Check 1: Determinant is not degenerate
        val det = homography[0] * (homography[4] * homography[8] - homography[5] * homography[7]) -
                homography[1] * (homography[3] * homography[8] - homography[5] * homography[6]) +
                homography[2] * (homography[3] * homography[7] - homography[4] * homography[6])

        if (Math.abs(det) < 1e-10f) return false

        // Check 2: Reprojection error
        val error = projector.computeReprojectionError(detectedKeypoints, homography)
        if (error > MAX_REPROJECTION_ERROR || error.isNaN()) return false

        return true
    }

    /**
     * Filter projected keypoints by geometric validity.
     *
     * Near-court points (KP9-16) are always kept (they come from direct detection).
     * Far-court points (KP1-8) are validated against relaxed geometric constraints:
     * - Must be within image bounds (30% margin)
     * - Must be above (lower y) the near-court service line (min Y) with tolerance
     * - Far baseline must be narrower than near baseline (perspective foreshortening)
     * - Far-court cannot extend too far above near-court
     *
     * Invalid far-court points are removed to prevent wild overlay rendering.
     */
    fun filterValidProjections(
        projected: List<Keypoint>,
        detected: List<Keypoint>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Keypoint> {
        if (projected.isEmpty()) return projected

        // Separate near-court (KP9-16) and far-court (KP1-8)
        val nearCourt = projected.filter { it.id in 9..16 }
        val farCourt = projected.filter { it.id in 1..8 }

        // Always keep near-court points (from direct detection, reliable)
        val result = nearCourt.toMutableList()

        if (farCourt.isEmpty() || nearCourt.isEmpty()) return result

        // Reference: use service line Y (KP9-11) for "top of near court"
        val nearYs = detected.filter { it.id in 9..16 }.map { it.y }
        val nearMinY = nearYs.minOrNull() ?: return result
        val nearMaxY = nearYs.maxOrNull() ?: return result
        val nearCourtHeight = nearMaxY - nearMinY

        // Near baseline width (KP12 to KP16)
        val nearLeft = detected.find { it.id == 12 }
        val nearRight = detected.find { it.id == 16 }
        val nearBaselineWidth = if (nearLeft != null && nearRight != null) {
            abs(nearRight.x - nearLeft.x)
        } else {
            imageWidth * 0.5f
        }

        // Far baseline width (KP1 to KP5)
        val farLeft = farCourt.find { it.id == 1 }
        val farRight = farCourt.find { it.id == 5 }
        val farBaselineWidth = if (farLeft != null && farRight != null) {
            abs(farRight.x - farLeft.x)
        } else {
            null
        }

        // Geometric check: far baseline should be narrower than near baseline
        val farNearRatio = if (farBaselineWidth != null && nearBaselineWidth > 0) {
            farBaselineWidth / nearBaselineWidth
        } else {
            null
        }

        // Relaxed ratio: 0.05~0.90
        val farCourtGeometryValid = farNearRatio == null || farNearRatio in 0.05f..0.90f

        // Max allowed vertical extent above near-court top (relaxed: 2.5x)
        val maxFarExtent = if (nearCourtHeight > 0) nearCourtHeight * 2.5f else imageHeight * 0.6f

        // Relaxed margin: 30% of image width
        val margin = imageWidth * 0.3f

        // Allow far-court points slightly below near-court service line
        // (perspective can push far-court points close to or slightly past service line)
        val yTolerance = if (nearCourtHeight > 0) nearCourtHeight * 0.15f else 20f

        // Validate each far-court point
        var validFarCount = 0
        val farCandidates = mutableListOf<Keypoint>()

        for (kp in farCourt) {
            var valid = true

            // Bound check: relaxed bounds
            if (kp.x < -margin || kp.x > imageWidth + margin ||
                kp.y < -margin || kp.y > imageHeight + margin) {
                valid = false
            }

            // Far-court points must be above near-court top (with tolerance)
            if (kp.y > nearMinY + yTolerance) {
                valid = false
            }

            // Far-court points cannot extend too far above near-court
            if (kp.y < nearMinY - maxFarExtent) {
                valid = false
            }

            // Overall geometry must be valid
            if (!farCourtGeometryValid) {
                valid = false
            }

            if (valid) {
                farCandidates.add(kp)
                validFarCount++
            }
        }

        // Relaxed: at least 4 of 8 far-court points must pass (was 5)
        if (validFarCount >= 4) {
            result.addAll(farCandidates)
        }

        return result
    }
}
