package com.shot.app.ui

import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import com.shot.app.viewmodel.CameraViewModel
import com.shot.core.ItfCourtSpec
import com.shot.core.R
import com.shot.core.model.ConfidenceLevel
import com.shot.core.model.DetectionStatus
import com.shot.core.model.Keypoint
import com.shot.detection.BallTrackingDetector

// --- Color System ---
private object ShotColors {
    val surface = Color(0xFF0D0D0D)
    val surfaceOverlay = Color(0xFF0D0D0D).copy(alpha = 0.65f)
    val surfaceGlass = Color(0xFF1A1A1A).copy(alpha = 0.72f)
    val glassBorder = Color.White.copy(alpha = 0.08f)
    val glassInnerBorder = Color.White.copy(alpha = 0.04f)

    val courtLine = Color(0xFF00E676)      // Vivid green for court lines
    val courtLineFar = Color(0xFF00E676).copy(alpha = 0.45f) // Faded for far court
    val kpHigh = Color(0xFF00E676)
    val kpMedium = Color(0xFFFFD740)
    val kpLow = Color(0xFFFF5252)

    val statusGreen = Color(0xFF00E676)
    val statusYellow = Color(0xFFFFD740)
    val statusRed = Color(0xFFFF5252)

    val textPrimary = Color(0xFFF5F5F5)
    val textSecondary = Color(0xFF9E9E9E)
    val textMono = Color(0xFFB0BEC5)

    val accentTeal = Color(0xFF00BFA5)
}

@Composable
fun CameraScreen(
    viewModel: CameraViewModel = hiltViewModel()
) {
    val detectionResult by viewModel.detectionResult.collectAsState()
    val ballPosition by viewModel.ballPosition.collectAsState()
    val isDebugMode by viewModel.isDebugMode.collectAsState()

    // FPS calculation
    var fps by remember { mutableFloatStateOf(0f) }
    var frameCount by remember { mutableLongStateOf(0L) }
    var fpsUpdateTime by remember { mutableLongStateOf(System.currentTimeMillis()) }

    LaunchedEffect(detectionResult) {
        frameCount++
        val now = System.currentTimeMillis()
        val elapsed = now - fpsUpdateTime
        if (elapsed >= 1000) {
            fps = frameCount * 1000f / elapsed
            frameCount = 0
            fpsUpdateTime = now
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Layer 1: Camera Preview
        AndroidView(
            factory = { context ->
                PreviewView(context).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            update = { previewView ->
                viewModel.bindCamera(previewView)
            },
            modifier = Modifier.fillMaxSize()
        )

        // Layer 2: Court Line Overlay
        CourtOverlay(
            projectedKeypoints = detectionResult.projectedKeypoints,
            detectedKeypoints = detectionResult.detectedKeypoints,
            isDebugMode = isDebugMode
        )

        // Layer 2.5: Ball Position Overlay
        BallOverlay(
            ballPosition = ballPosition,
            isDebugMode = isDebugMode
        )

        // Layer 3: Top bar — Status pill (Dynamic Island style)
        StatusPill(
            status = detectionResult.status,
            fps = fps,
            isDebugMode = isDebugMode,
            onToggleDebug = { viewModel.toggleDebugMode() },
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 12.dp)
        )

        // Layer 4: Debug panel (bottom-left)
        AnimatedVisibility(
            visible = isDebugMode,
            enter = fadeIn(tween(200)),
            exit = fadeOut(tween(150)),
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
        ) {
            DebugPanel(result = detectionResult, fps = fps, ballPosition = ballPosition)
        }

        // Layer 5: Camera guide (centered, when not detected)
        AnimatedVisibility(
            visible = detectionResult.status == DetectionStatus.NOT_DETECTED,
            enter = fadeIn(tween(400)),
            exit = fadeOut(tween(200)),
            modifier = Modifier.align(Alignment.Center)
        ) {
            CameraGuide()
        }

        // Layer 6: Bottom bar
        BottomBar(
            isDebugMode = isDebugMode,
            onToggleDebug = { viewModel.toggleDebugMode() },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp)
        )
    }
}

// --- Status Pill (Dynamic Island Style) ---
@Composable
private fun StatusPill(
    status: DetectionStatus,
    fps: Float,
    isDebugMode: Boolean,
    onToggleDebug: () -> Unit,
    modifier: Modifier = Modifier
) {
    val (statusColor, statusText) = when (status) {
        DetectionStatus.DETECTED -> ShotColors.statusGreen to stringResource(R.string.court_detected)
        DetectionStatus.PARTIAL -> ShotColors.statusYellow to stringResource(R.string.court_partial)
        DetectionStatus.NOT_DETECTED -> ShotColors.statusRed to stringResource(R.string.court_not_detected)
    }

    val pillAlpha by animateFloatAsState(
        targetValue = if (status == DetectionStatus.DETECTED) 0.85f else 0.92f,
        animationSpec = tween(300),
        label = "pillAlpha"
    )

    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = modifier
            .background(
                ShotColors.surface.copy(alpha = pillAlpha),
                RoundedCornerShape(20.dp)
            )
            .border(1.dp, ShotColors.glassBorder, RoundedCornerShape(20.dp))
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null
            ) { onToggleDebug() }
            .padding(horizontal = 14.dp, vertical = 7.dp)
    ) {
        // Status dot
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(statusColor, CircleShape)
        )
        Spacer(modifier = Modifier.width(8.dp))

        // Status text
        Text(
            text = statusText,
            color = ShotColors.textPrimary,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.3.sp
        )

        if (isDebugMode) {
            Spacer(modifier = Modifier.width(10.dp))
            // FPS indicator
            Text(
                text = "${"%.0f".format(fps)} fps",
                color = ShotColors.textMono,
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 0.5.sp
            )
        }
    }
}

// --- Court Line Overlay ---
@Composable
private fun CourtOverlay(
    projectedKeypoints: List<Keypoint>,
    detectedKeypoints: List<Keypoint>,
    isDebugMode: Boolean
) {
    if (projectedKeypoints.isEmpty()) return

    val keypointMap = projectedKeypoints.associateBy { it.id }

    // Near court IDs (directly detected)
    val nearCourtIds = setOf(9, 10, 11, 12, 13, 14, 15, 16)

    Canvas(modifier = Modifier.fillMaxSize()) {
        val imageAspect = 1280f / 720f
        val canvasAspect = size.width / size.height

        val scaleX: Float
        val scaleY: Float
        val offsetX: Float
        val offsetY: Float

        if (imageAspect > canvasAspect) {
            scaleY = size.height / 720f
            scaleX = scaleY
            offsetX = (size.width - 1280f * scaleX) / 2f
            offsetY = 0f
        } else {
            scaleX = size.width / 1280f
            scaleY = scaleX
            offsetX = 0f
            offsetY = (size.height - 720f * scaleY) / 2f
        }

        fun toScreen(kp: Keypoint) = Offset(
            kp.x * scaleX + offsetX,
            kp.y * scaleY + offsetY
        )

        // Draw court lines with near/far distinction
        for ((startId, endId) in ItfCourtSpec.COURT_LINES) {
            val start = keypointMap[startId] ?: continue
            val end = keypointMap[endId] ?: continue

            // Near court lines are brighter, far court lines are faded
            val isNearLine = startId in nearCourtIds && endId in nearCourtIds
            val lineColor = if (isNearLine) ShotColors.courtLine else ShotColors.courtLineFar
            val lineWidth = if (isNearLine) 2.5f else 1.5f

            // Far court lines use dashed style
            val pathEffect = if (!isNearLine) {
                PathEffect.dashPathEffect(floatArrayOf(12f, 8f), 0f)
            } else null

            drawLine(
                color = lineColor,
                start = toScreen(start),
                end = toScreen(end),
                strokeWidth = lineWidth,
                cap = StrokeCap.Round,
                pathEffect = pathEffect
            )
        }

        // Draw keypoints
        for (kp in detectedKeypoints) {
            val center = toScreen(kp)
            val color = when (ConfidenceLevel.from(kp.confidence)) {
                ConfidenceLevel.HIGH -> ShotColors.kpHigh
                ConfidenceLevel.MEDIUM -> ShotColors.kpMedium
                ConfidenceLevel.LOW -> ShotColors.kpLow
            }

            // Outer glow ring
            drawCircle(
                color = color.copy(alpha = 0.2f),
                radius = 14f,
                center = center
            )
            // Solid dot
            drawCircle(
                color = color,
                radius = 5f,
                center = center
            )
            // White border for contrast
            drawCircle(
                color = Color.White.copy(alpha = 0.6f),
                radius = 5f,
                center = center,
                style = Stroke(width = 1.2f)
            )

            if (isDebugMode) {
                // Confidence ring
                drawCircle(
                    color = color.copy(alpha = 0.5f),
                    radius = 10f,
                    center = center,
                    style = Stroke(width = 1.5f)
                )
            }
        }
    }
}

// --- Ball Position Overlay ---
@Composable
private fun BallOverlay(
    ballPosition: BallTrackingDetector.BallPosition?,
    isDebugMode: Boolean
) {
    if (ballPosition == null || !ballPosition.detected) return

    Canvas(modifier = Modifier.fillMaxSize()) {
        val imageAspect = 1280f / 720f
        val canvasAspect = size.width / size.height

        val scaleX: Float
        val scaleY: Float
        val offsetX: Float
        val offsetY: Float

        if (imageAspect > canvasAspect) {
            scaleY = size.height / 720f
            scaleX = scaleY
            offsetX = (size.width - 1280f * scaleX) / 2f
            offsetY = 0f
        } else {
            scaleX = size.width / 1280f
            scaleY = scaleX
            offsetX = 0f
            offsetY = (size.height - 720f * scaleY) / 2f
        }

        val center = Offset(
            ballPosition.x * scaleX + offsetX,
            ballPosition.y * scaleY + offsetY
        )

        val ballColor = Color(0xFFFF6D00) // Vivid orange for ball

        // Outer glow pulse
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    ballColor.copy(alpha = 0.4f),
                    ballColor.copy(alpha = 0f)
                ),
                center = center,
                radius = 28f
            ),
            radius = 28f,
            center = center
        )

        // Core dot
        drawCircle(
            color = ballColor,
            radius = 7f,
            center = center
        )

        // White border ring
        drawCircle(
            color = Color.White.copy(alpha = 0.8f),
            radius = 7f,
            center = center,
            style = Stroke(width = 1.8f)
        )

        // Crosshair lines (subtle)
        val crossLen = 14f
        val crossColor = ballColor.copy(alpha = 0.6f)
        drawLine(crossColor, Offset(center.x - crossLen, center.y), Offset(center.x - 9f, center.y), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x + 9f, center.y), Offset(center.x + crossLen, center.y), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x, center.y - crossLen), Offset(center.x, center.y - 9f), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x, center.y + 9f), Offset(center.x, center.y + crossLen), strokeWidth = 1.2f, cap = StrokeCap.Round)

        if (isDebugMode) {
            // Confidence ring scaled by confidence value
            val confRadius = 12f + ballPosition.confidence * 8f
            drawCircle(
                color = ballColor.copy(alpha = 0.35f),
                radius = confRadius,
                center = center,
                style = Stroke(width = 1.5f)
            )
        }
    }
}

// --- Camera Guide ---
@Composable
private fun CameraGuide() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .background(ShotColors.surfaceGlass, RoundedCornerShape(16.dp))
            .border(1.dp, ShotColors.glassBorder, RoundedCornerShape(16.dp))
            .padding(horizontal = 28.dp, vertical = 20.dp)
    ) {
        // Court icon (simple lines)
        Canvas(modifier = Modifier.size(48.dp)) {
            val w = size.width
            val h = size.height
            val lineColor = ShotColors.textSecondary
            val stroke = 1.5f

            // Court outline
            drawRect(color = lineColor, style = Stroke(stroke))
            // Net line
            drawLine(lineColor, Offset(0f, h / 2), Offset(w, h / 2), stroke)
            // Service lines
            drawLine(lineColor, Offset(w * 0.2f, h * 0.25f), Offset(w * 0.8f, h * 0.25f), stroke)
            drawLine(lineColor, Offset(w * 0.2f, h * 0.75f), Offset(w * 0.8f, h * 0.75f), stroke)
            // Center line
            drawLine(lineColor, Offset(w / 2, h * 0.25f), Offset(w / 2, h * 0.75f), stroke)
        }

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = stringResource(R.string.camera_guide_center),
            color = ShotColors.textPrimary,
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.2.sp
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Position your camera behind the baseline",
            color = ShotColors.textSecondary,
            fontSize = 12.sp,
            letterSpacing = 0.1.sp
        )
    }
}

// --- Debug Panel ---
@Composable
private fun DebugPanel(
    result: com.shot.core.model.CourtDetectionResult,
    fps: Float,
    ballPosition: BallTrackingDetector.BallPosition?,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(ShotColors.surfaceGlass, RoundedCornerShape(12.dp))
            .border(1.dp, ShotColors.glassBorder, RoundedCornerShape(12.dp))
            .padding(10.dp)
    ) {
        DebugRow("FPS", "${"%.1f".format(fps)}")
        DebugRow("Inference", "${result.inferenceTimeMs} ms")
        DebugRow("Reproj", "${"%.1f".format(result.reprojectionError)} px")
        DebugRow("Keypoints", "${result.reliableKeypointCount}/8")
        DebugRow("Projected", "${result.projectedKeypoints.size}/16")
        DebugRow("Status", result.status.name)

        // Ball tracking info
        Spacer(modifier = Modifier.height(4.dp))
        if (ballPosition != null) {
            DebugRow("Ball", if (ballPosition.detected) "DETECTED" else "---")
            DebugRow("Ball Conf", "${"%.2f".format(ballPosition.confidence)}")
            if (ballPosition.detected) {
                DebugRow("Ball Pos", "${"%.0f".format(ballPosition.x)}, ${"%.0f".format(ballPosition.y)}")
            }
        } else {
            DebugRow("Ball", "INIT...")
        }
    }
}

@Composable
private fun DebugRow(label: String, value: String) {
    Row(
        modifier = Modifier.padding(vertical = 1.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            color = ShotColors.textSecondary,
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(72.dp)
        )
        Text(
            text = value,
            color = ShotColors.textMono,
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium
        )
    }
}

// --- Bottom Bar ---
@Composable
private fun BottomBar(
    isDebugMode: Boolean,
    onToggleDebug: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        horizontalArrangement = Arrangement.Center,
        modifier = modifier
    ) {
        // Debug toggle chip
        Text(
            text = if (isDebugMode) "DEBUG" else "DEBUG",
            color = if (isDebugMode) ShotColors.accentTeal else ShotColors.textSecondary,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp,
            modifier = Modifier
                .background(
                    if (isDebugMode) ShotColors.accentTeal.copy(alpha = 0.12f)
                    else ShotColors.surfaceGlass,
                    RoundedCornerShape(8.dp)
                )
                .border(
                    1.dp,
                    if (isDebugMode) ShotColors.accentTeal.copy(alpha = 0.3f)
                    else ShotColors.glassBorder,
                    RoundedCornerShape(8.dp)
                )
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) { onToggleDebug() }
                .padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}
