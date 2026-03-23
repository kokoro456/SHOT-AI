package com.shot.core

import android.graphics.PointF

/**
 * ITF standard tennis court dimensions and keypoint positions.
 *
 * Reference: ITF Rules of Tennis 2024, Section 1 (Court and Equipment).
 *
 * Coordinate system:
 * - Origin at keypoint 12 (near baseline × doubles left sideline)
 * - X-axis: along the baseline (left to right facing the net)
 * - Y-axis: toward the net and far baseline
 * - All units in meters
 */
object ItfCourtSpec {

    // Court dimensions (meters)
    const val COURT_LENGTH = 23.77f       // Total length baseline to baseline
    const val SINGLES_WIDTH = 8.23f       // Singles court width
    const val DOUBLES_WIDTH = 10.97f      // Doubles court width
    const val SERVICE_LINE_DIST = 6.40f   // Service line distance from baseline
    const val NET_DIST = 11.885f          // Net distance from baseline (half court length)
    const val LINE_WIDTH_MIN = 0.025f     // Minimum line width (2.5cm)
    const val LINE_WIDTH_MAX = 0.05f      // Maximum line width (5cm)
    const val NET_HEIGHT_CENTER = 0.914f  // Net height at center (3ft)

    // Derived dimensions
    const val DOUBLES_ALLEY = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2f  // 1.37m

    /**
     * All 16 keypoint positions in court coordinates (meters).
     * Origin at keypoint 12 (near baseline × doubles left sideline).
     *
     * Court layout (viewed from above):
     * ```
     *  1─────2────────────3───────────────4─────5   ← Far baseline (y = COURT_LENGTH)
     *  |     6────────────7───────────────8     |   ← Far service line
     *  |     |          [NET]             |     |   ← Net (y = NET_DIST)
     *  |     9───────────10──────────────11     |   ← Near service line
     * 12────13───────────14──────────────15────16   ← Near baseline (y = 0)
     * ```
     */
    val KEYPOINTS: Map<Int, PointF> = mapOf(
        // Near baseline (y = 0)
        12 to PointF(0f, 0f),
        13 to PointF(DOUBLES_ALLEY, 0f),
        14 to PointF(DOUBLES_WIDTH / 2f, 0f),
        15 to PointF(DOUBLES_ALLEY + SINGLES_WIDTH, 0f),
        16 to PointF(DOUBLES_WIDTH, 0f),

        // Near service line (y = SERVICE_LINE_DIST)
        9  to PointF(DOUBLES_ALLEY, SERVICE_LINE_DIST),
        10 to PointF(DOUBLES_WIDTH / 2f, SERVICE_LINE_DIST),
        11 to PointF(DOUBLES_ALLEY + SINGLES_WIDTH, SERVICE_LINE_DIST),

        // Far service line (y = COURT_LENGTH - SERVICE_LINE_DIST)
        6  to PointF(DOUBLES_ALLEY, COURT_LENGTH - SERVICE_LINE_DIST),
        7  to PointF(DOUBLES_WIDTH / 2f, COURT_LENGTH - SERVICE_LINE_DIST),
        8  to PointF(DOUBLES_ALLEY + SINGLES_WIDTH, COURT_LENGTH - SERVICE_LINE_DIST),

        // Far baseline (y = COURT_LENGTH)
        1  to PointF(0f, COURT_LENGTH),
        2  to PointF(DOUBLES_ALLEY, COURT_LENGTH),
        3  to PointF(DOUBLES_WIDTH / 2f, COURT_LENGTH),
        4  to PointF(DOUBLES_ALLEY + SINGLES_WIDTH, COURT_LENGTH),
        5  to PointF(DOUBLES_WIDTH, COURT_LENGTH),
    )

    /**
     * Court line segments defined as pairs of keypoint IDs.
     * Used for drawing the court overlay.
     */
    val COURT_LINES: List<Pair<Int, Int>> = listOf(
        // Baselines
        12 to 16, // Near baseline (doubles)
        1 to 5,   // Far baseline (doubles)

        // Doubles sidelines
        12 to 1,  // Left doubles sideline
        16 to 5,  // Right doubles sideline

        // Singles sidelines
        13 to 2,  // Left singles sideline
        15 to 4,  // Right singles sideline

        // Service lines
        9 to 11,  // Near service line
        6 to 8,   // Far service line

        // Center service line
        10 to 7,  // Center service line (between service lines)

        // Center marks (short marks on baselines - represented as points 14 and 3)
    )

    /**
     * Get singles court lines only (excluding doubles alleys).
     */
    val SINGLES_LINES: List<Pair<Int, Int>> = listOf(
        13 to 15, // Near baseline (singles)
        2 to 4,   // Far baseline (singles)
        13 to 2,  // Left singles sideline
        15 to 4,  // Right singles sideline
        9 to 11,  // Near service line
        6 to 8,   // Far service line
        10 to 7,  // Center service line
    )

    /** Check if court coordinate is within doubles court boundaries */
    fun isInDoubles(courtX: Float, courtY: Float): Boolean {
        return courtX >= 0f && courtX <= DOUBLES_WIDTH &&
               courtY >= 0f && courtY <= COURT_LENGTH
    }

    /** Check if court coordinate is within singles court boundaries */
    fun isInSingles(courtX: Float, courtY: Float): Boolean {
        val singlesLeft = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2f
        val singlesRight = singlesLeft + SINGLES_WIDTH
        return courtX >= singlesLeft && courtX <= singlesRight &&
               courtY >= 0f && courtY <= COURT_LENGTH
    }

    /** Check if court coordinate is in (any part of ball touching line counts as in) */
    fun isIn(courtX: Float, courtY: Float, ballRadiusM: Float = 0.0335f): Boolean {
        return courtX >= -ballRadiusM && courtX <= DOUBLES_WIDTH + ballRadiusM &&
               courtY >= -ballRadiusM && courtY <= COURT_LENGTH + ballRadiusM
    }
}
