package com.shot.court

import com.shot.core.ItfCourtSpec
import com.shot.core.model.Keypoint

/**
 * Computes the homography matrix that maps image pixel coordinates to
 * ITF court coordinates (meters), and vice versa.
 *
 * Uses the Direct Linear Transform (DLT) algorithm with confidence-weighted
 * SVD via Jacobi iteration (500 sweeps for numerical stability).
 */
class HomographyCalculator {

    fun computeHomography(detectedKeypoints: List<Keypoint>): FloatArray? {
        val validKeypoints = detectedKeypoints.filter { it.confidence > 0.05f }
        if (validKeypoints.size < 4) return null

        val pairs = mutableListOf<Triple<FloatArray, FloatArray, Float>>()
        for (kp in validKeypoints) {
            val courtPoint = ItfCourtSpec.KEYPOINTS[kp.id] ?: continue
            pairs.add(
                Triple(
                    floatArrayOf(kp.x, kp.y),
                    floatArrayOf(courtPoint.x, courtPoint.y),
                    kp.confidence.coerceAtLeast(0.1f)
                )
            )
        }

        if (pairs.size < 4) return null
        return computeDLT(pairs)
    }

    fun projectCourtToImage(courtX: Float, courtY: Float, hInv: FloatArray): FloatArray {
        val w = hInv[6] * courtX + hInv[7] * courtY + hInv[8]
        if (Math.abs(w) < 1e-10f) return floatArrayOf(Float.NaN, Float.NaN)
        val x = (hInv[0] * courtX + hInv[1] * courtY + hInv[2]) / w
        val y = (hInv[3] * courtX + hInv[4] * courtY + hInv[5]) / w
        return floatArrayOf(x, y)
    }

    fun projectImageToCourt(imageX: Float, imageY: Float, h: FloatArray): FloatArray {
        val w = h[6] * imageX + h[7] * imageY + h[8]
        if (Math.abs(w) < 1e-10f) return floatArrayOf(Float.NaN, Float.NaN)
        val x = (h[0] * imageX + h[1] * imageY + h[2]) / w
        val y = (h[3] * imageX + h[4] * imageY + h[5]) / w
        return floatArrayOf(x, y)
    }

    fun invertHomography(h: FloatArray): FloatArray? {
        val det = h[0].toDouble() * (h[4].toDouble() * h[8].toDouble() - h[5].toDouble() * h[7].toDouble()) -
                h[1].toDouble() * (h[3].toDouble() * h[8].toDouble() - h[5].toDouble() * h[6].toDouble()) +
                h[2].toDouble() * (h[3].toDouble() * h[7].toDouble() - h[4].toDouble() * h[6].toDouble())

        if (Math.abs(det) < 1e-10) return null

        val invDet = 1.0 / det
        return floatArrayOf(
            ((h[4].toDouble() * h[8].toDouble() - h[5].toDouble() * h[7].toDouble()) * invDet).toFloat(),
            ((h[2].toDouble() * h[7].toDouble() - h[1].toDouble() * h[8].toDouble()) * invDet).toFloat(),
            ((h[1].toDouble() * h[5].toDouble() - h[2].toDouble() * h[4].toDouble()) * invDet).toFloat(),
            ((h[5].toDouble() * h[6].toDouble() - h[3].toDouble() * h[8].toDouble()) * invDet).toFloat(),
            ((h[0].toDouble() * h[8].toDouble() - h[2].toDouble() * h[6].toDouble()) * invDet).toFloat(),
            ((h[2].toDouble() * h[3].toDouble() - h[0].toDouble() * h[5].toDouble()) * invDet).toFloat(),
            ((h[3].toDouble() * h[7].toDouble() - h[4].toDouble() * h[6].toDouble()) * invDet).toFloat(),
            ((h[1].toDouble() * h[6].toDouble() - h[0].toDouble() * h[7].toDouble()) * invDet).toFloat(),
            ((h[0].toDouble() * h[4].toDouble() - h[1].toDouble() * h[3].toDouble()) * invDet).toFloat()
        )
    }

    private fun computeDLT(pairs: List<Triple<FloatArray, FloatArray, Float>>): FloatArray? {
        val n = pairs.size

        // Normalize source points (image coordinates)
        val srcMeanX = pairs.map { it.first[0].toDouble() }.average()
        val srcMeanY = pairs.map { it.first[1].toDouble() }.average()
        val srcAvgDist = pairs.map {
            Math.sqrt(
                (it.first[0] - srcMeanX) * (it.first[0] - srcMeanX) +
                        (it.first[1] - srcMeanY) * (it.first[1] - srcMeanY)
            )
        }.average()
        val srcScale = if (srcAvgDist < 1e-10) 1.0 else Math.sqrt(2.0) / srcAvgDist

        // Normalize destination points (court coordinates)
        val dstMeanX = pairs.map { it.second[0].toDouble() }.average()
        val dstMeanY = pairs.map { it.second[1].toDouble() }.average()
        val dstAvgDist = pairs.map {
            Math.sqrt(
                (it.second[0] - dstMeanX) * (it.second[0] - dstMeanX) +
                        (it.second[1] - dstMeanY) * (it.second[1] - dstMeanY)
            )
        }.average()
        val dstScale = if (dstAvgDist < 1e-10) 1.0 else Math.sqrt(2.0) / dstAvgDist

        // Build the 2N x 9 matrix A with confidence weighting
        val a = Array(n * 2) { DoubleArray(9) }

        for (i in 0 until n) {
            val sx = (pairs[i].first[0] - srcMeanX) * srcScale
            val sy = (pairs[i].first[1] - srcMeanY) * srcScale
            val dx = (pairs[i].second[0] - dstMeanX) * dstScale
            val dy = (pairs[i].second[1] - dstMeanY) * dstScale
            val w = Math.sqrt(pairs[i].third.toDouble()) // confidence weight

            a[2 * i][0] = -sx * w; a[2 * i][1] = -sy * w; a[2 * i][2] = -1.0 * w
            a[2 * i][3] = 0.0; a[2 * i][4] = 0.0; a[2 * i][5] = 0.0
            a[2 * i][6] = dx * sx * w; a[2 * i][7] = dx * sy * w; a[2 * i][8] = dx * w

            a[2 * i + 1][0] = 0.0; a[2 * i + 1][1] = 0.0; a[2 * i + 1][2] = 0.0
            a[2 * i + 1][3] = -sx * w; a[2 * i + 1][4] = -sy * w; a[2 * i + 1][5] = -1.0 * w
            a[2 * i + 1][6] = dy * sx * w; a[2 * i + 1][7] = dy * sy * w; a[2 * i + 1][8] = dy * w
        }

        // Compute A^T * A (9x9 symmetric)
        val ata = Array(9) { DoubleArray(9) }
        for (i in 0 until 9) {
            for (j in i until 9) {
                var sum = 0.0
                for (k in 0 until n * 2) {
                    sum += a[k][i] * a[k][j]
                }
                ata[i][j] = sum
                ata[j][i] = sum
            }
        }

        // Find eigenvector of smallest eigenvalue using Jacobi iteration
        val h = smallestEigenvectorJacobi(ata, 9) ?: return null

        // Denormalize: H = T_dst_inv * H_normalized * T_src
        val tSrc = doubleArrayOf(
            srcScale, 0.0, -srcMeanX * srcScale,
            0.0, srcScale, -srcMeanY * srcScale,
            0.0, 0.0, 1.0
        )
        val tDstInv = doubleArrayOf(
            1.0 / dstScale, 0.0, dstMeanX,
            0.0, 1.0 / dstScale, dstMeanY,
            0.0, 0.0, 1.0
        )

        val temp = multiply3x3d(h, tSrc)
        val result = multiply3x3d(tDstInv, temp)

        // Normalize so result[8] = 1
        val scale = result[8]
        if (Math.abs(scale) < 1e-15) return null

        return FloatArray(9) { (result[it] / scale).toFloat() }
    }

    /**
     * Jacobi eigenvalue algorithm for 9x9 symmetric matrix.
     * Returns eigenvector corresponding to the smallest eigenvalue.
     * Uses 500 iterations for robust convergence on ill-conditioned matrices.
     */
    private fun smallestEigenvectorJacobi(matrix: Array<DoubleArray>, n: Int): DoubleArray? {
        val a = Array(n) { matrix[it].copyOf() }
        val v = Array(n) { DoubleArray(n) }
        for (i in 0 until n) v[i][i] = 1.0

        for (iter in 0 until 500) {
            // Find largest off-diagonal element
            var maxVal = 0.0
            var p = 0
            var q = 1
            for (i in 0 until n) {
                for (j in i + 1 until n) {
                    val absVal = Math.abs(a[i][j])
                    if (absVal > maxVal) {
                        maxVal = absVal
                        p = i
                        q = j
                    }
                }
            }

            if (maxVal < 1e-15) break

            // Compute Jacobi rotation angle
            val theta = if (Math.abs(a[p][p] - a[q][q]) < 1e-15) {
                Math.PI / 4.0
            } else {
                0.5 * Math.atan2(2.0 * a[p][q], a[p][p] - a[q][q])
            }

            val c = Math.cos(theta)
            val s = Math.sin(theta)

            // Apply rotation to matrix A
            val app = c * c * a[p][p] + 2 * s * c * a[p][q] + s * s * a[q][q]
            val aqq = s * s * a[p][p] - 2 * s * c * a[p][q] + c * c * a[q][q]
            a[p][q] = 0.0
            a[q][p] = 0.0
            a[p][p] = app
            a[q][q] = aqq

            for (i in 0 until n) {
                if (i != p && i != q) {
                    val aip = c * a[i][p] + s * a[i][q]
                    val aiq = -s * a[i][p] + c * a[i][q]
                    a[i][p] = aip; a[p][i] = aip
                    a[i][q] = aiq; a[q][i] = aiq
                }
            }

            // Update eigenvectors
            for (i in 0 until n) {
                val vip = c * v[i][p] + s * v[i][q]
                val viq = -s * v[i][p] + c * v[i][q]
                v[i][p] = vip
                v[i][q] = viq
            }
        }

        // Condition number check: reject ill-conditioned systems
        var minEig = Math.abs(a[0][0])
        var maxEig = Math.abs(a[0][0])
        var minIdx = 0
        for (i in 1 until n) {
            val absEig = Math.abs(a[i][i])
            if (absEig < minEig) { minEig = absEig; minIdx = i }
            if (absEig > maxEig) { maxEig = absEig }
        }

        // Reject if condition number is too high (ill-conditioned)
        if (minEig > 0 && maxEig / minEig > 1e8) return null

        return DoubleArray(n) { v[it][minIdx] }
    }

    private fun multiply3x3d(a: DoubleArray, b: DoubleArray): DoubleArray {
        return doubleArrayOf(
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
        )
    }
}
