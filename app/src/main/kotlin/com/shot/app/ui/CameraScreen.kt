package com.shot.app.ui

import android.app.Activity
import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.PixelCopy
import android.widget.Toast
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
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
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// --- Color System ---
private object ShotColors {
    val surface = Color(0xFF0D0D0D)
    val surfaceOverlay = Color(0xFF0D0D0D).copy(alpha = 0.65f)
    val surfaceGlass = Color(0xFF1A1A1A).copy(alpha = 0.72f)
    val glassBorder = Color.White.copy(alpha = 0.08f)
    val glassInnerBorder = Color.White.copy(alpha = 0.04f)

    // Neon fluorescent court lines
    val courtLine = Color(0xFF39FF14)          // Electric green (neon)
    val courtLineGlow = Color(0xFF39FF14).copy(alpha = 0.35f)  // Glow layer
    val courtLineFar = Color(0xFF39FF14).copy(alpha = 0.65f)   // Far court (brighter than before)
    val courtLineFarGlow = Color(0xFF39FF14).copy(alpha = 0.20f)

    val kpHigh = Color(0xFF39FF14)
    val kpMedium = Color(0xFFFFD740)
    val kpLow = Color(0xFFFF5252)

    val statusGreen = Color(0xFF39FF14)
    val statusYellow = Color(0xFFFFD740)
    val statusRed = Color(0xFFFF5252)

    val textPrimary = Color(0xFFF5F5F5)
    val textSecondary = Color(0xFF9E9E9E)
    val textMono = Color(0xFFB0BEC5)

    val accentTeal = Color(0xFF00BFA5)
    val lockBlue = Color(0xFF42A5F5)
    val alertOrange = Color(0xFFFF9800)
    val captureWhite = Color(0xFFFFFFFF)
}

@Composable
fun CameraScreen(
    viewModel: CameraViewModel = hiltViewModel()
) {
    val detectionResult by viewModel.detectionResult.collectAsState()
    val ballPosition by viewModel.ballPosition.collectAsState()
    val isDebugMode by viewModel.isDebugMode.collectAsState()
    val isCourtLocked by viewModel.isCourtLockedFlow.collectAsState()
    val cameraMovedAlert by viewModel.cameraMovedAlert.collectAsState()

    val context = LocalContext.current
    val view = LocalView.current

    // Screenshot flash effect
    var showFlash by remember { mutableStateOf(false) }
    LaunchedEffect(showFlash) {
        if (showFlash) {
            kotlinx.coroutines.delay(150)
            showFlash = false
        }
    }

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
            factory = { ctx ->
                PreviewView(ctx).apply {
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
            isDebugMode = isDebugMode,
            isLocked = isCourtLocked
        )

        // Layer 2.5: Ball Position Overlay
        BallOverlay(
            ballPosition = ballPosition,
            isDebugMode = isDebugMode
        )

        // Layer 3: Top bar — Status pill
        StatusPill(
            status = detectionResult.status,
            fps = fps,
            isDebugMode = isDebugMode,
            isCourtLocked = isCourtLocked,
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
            DebugPanel(
                result = detectionResult,
                fps = fps,
                ballPosition = ballPosition,
                isCourtLocked = isCourtLocked
            )
        }

        // Layer 5: Camera guide (centered, when not detected)
        AnimatedVisibility(
            visible = detectionResult.status == DetectionStatus.NOT_DETECTED && !isCourtLocked,
            enter = fadeIn(tween(400)),
            exit = fadeOut(tween(200)),
            modifier = Modifier.align(Alignment.Center)
        ) {
            CameraGuide()
        }

        // Layer 5.5: Camera moved alert
        AnimatedVisibility(
            visible = cameraMovedAlert,
            enter = fadeIn(tween(300)),
            exit = fadeOut(tween(200)),
            modifier = Modifier.align(Alignment.Center)
        ) {
            CameraMovedAlert(
                onDismiss = { viewModel.dismissMovementAlert() }
            )
        }

        // Screenshot flash overlay
        if (showFlash) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White.copy(alpha = 0.6f))
            )
        }

        // Layer 6: Bottom bar
        BottomBar(
            isDebugMode = isDebugMode,
            isCourtLocked = isCourtLocked,
            canLock = detectionResult.status != DetectionStatus.NOT_DETECTED,
            onToggleDebug = { viewModel.toggleDebugMode() },
            onToggleLock = {
                if (isCourtLocked) viewModel.unlockCourt() else viewModel.lockCourt()
            },
            onCapture = {
                showFlash = true
                captureScreenshot(context as Activity, view)
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp)
        )
    }
}

// --- Screenshot Capture ---
private fun captureScreenshot(activity: Activity, view: android.view.View) {
    val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        PixelCopy.request(
            activity.window,
            bitmap,
            { result ->
                if (result == PixelCopy.SUCCESS) {
                    saveBitmapToGallery(activity, bitmap)
                } else {
                    Toast.makeText(activity, "Screenshot failed", Toast.LENGTH_SHORT).show()
                }
            },
            Handler(Looper.getMainLooper())
        )
    } else {
        // Fallback for older devices
        view.isDrawingCacheEnabled = true
        @Suppress("DEPRECATION")
        val cache = view.drawingCache
        if (cache != null) {
            saveBitmapToGallery(activity, cache.copy(Bitmap.Config.ARGB_8888, false))
        }
        view.isDrawingCacheEnabled = false
    }
}

private fun saveBitmapToGallery(activity: Activity, bitmap: Bitmap) {
    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val filename = "SHOT_$timestamp.jpg"

    val contentValues = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SHOT")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
    }

    val resolver = activity.contentResolver
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

    if (uri != null) {
        resolver.openOutputStream(uri)?.use { stream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.clear()
            contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, contentValues, null, null)
        }
        Toast.makeText(activity, "Saved: $filename", Toast.LENGTH_SHORT).show()
    } else {
        Toast.makeText(activity, "Save failed", Toast.LENGTH_SHORT).show()
    }
}

// --- Status Pill (Dynamic Island Style) ---
@Composable
private fun StatusPill(
    status: DetectionStatus,
    fps: Float,
    isDebugMode: Boolean,
    isCourtLocked: Boolean,
    onToggleDebug: () -> Unit,
    modifier: Modifier = Modifier
) {
    val (statusColor, statusText) = when {
        isCourtLocked -> ShotColors.lockBlue to "LOCKED"
        status == DetectionStatus.DETECTED -> ShotColors.statusGreen to stringResource(R.string.court_detected)
        status == DetectionStatus.PARTIAL -> ShotColors.statusYellow to stringResource(R.string.court_partial)
        else -> ShotColors.statusRed to stringResource(R.string.court_not_detected)
    }

    val pillAlpha by animateFloatAsState(
        targetValue = if (status == DetectionStatus.DETECTED || isCourtLocked) 0.85f else 0.92f,
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
            .border(
                1.dp,
                if (isCourtLocked) ShotColors.lockBlue.copy(alpha = 0.3f) else ShotColors.glassBorder,
                RoundedCornerShape(20.dp)
            )
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null
            ) { onToggleDebug() }
            .padding(horizontal = 14.dp, vertical = 7.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(statusColor, CircleShape)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = statusText,
            color = ShotColors.textPrimary,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.3.sp
        )
        if (isDebugMode) {
            Spacer(modifier = Modifier.width(10.dp))
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
    isDebugMode: Boolean,
    isLocked: Boolean = false
) {
    if (projectedKeypoints.isEmpty()) return

    val keypointMap = projectedKeypoints.associateBy { it.id }
    val nearCourtIds = setOf(9, 10, 11, 12, 13, 14, 15, 16)

    // When locked, use blue neon; otherwise electric green neon
    val nearColor = if (isLocked) ShotColors.lockBlue else ShotColors.courtLine
    val nearGlow = if (isLocked) ShotColors.lockBlue.copy(alpha = 0.30f) else ShotColors.courtLineGlow
    val farColor = if (isLocked) ShotColors.lockBlue.copy(alpha = 0.65f) else ShotColors.courtLineFar
    val farGlow = if (isLocked) ShotColors.lockBlue.copy(alpha = 0.15f) else ShotColors.courtLineFarGlow

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

        for ((startId, endId) in ItfCourtSpec.COURT_LINES) {
            val start = keypointMap[startId] ?: continue
            val end = keypointMap[endId] ?: continue

            val isNearLine = startId in nearCourtIds && endId in nearCourtIds
            val lineColor = if (isNearLine) nearColor else farColor
            val glowColor = if (isNearLine) nearGlow else farGlow

            // All lines solid, near thicker than far
            val coreWidth = if (isNearLine) 4.0f else 2.5f
            val glowWidth = if (isNearLine) 12f else 8f

            val startPt = toScreen(start)
            val endPt = toScreen(end)

            // Layer 1: Wide glow (neon bloom effect)
            drawLine(
                color = glowColor,
                start = startPt,
                end = endPt,
                strokeWidth = glowWidth,
                cap = StrokeCap.Round
            )

            // Layer 2: Core line (bright center)
            drawLine(
                color = lineColor,
                start = startPt,
                end = endPt,
                strokeWidth = coreWidth,
                cap = StrokeCap.Round
            )
        }

        // Draw keypoints (skip when locked)
        if (!isLocked) {
            for (kp in detectedKeypoints) {
                val center = toScreen(kp)
                val color = when (ConfidenceLevel.from(kp.confidence)) {
                    ConfidenceLevel.HIGH -> ShotColors.kpHigh
                    ConfidenceLevel.MEDIUM -> ShotColors.kpMedium
                    ConfidenceLevel.LOW -> ShotColors.kpLow
                }

                drawCircle(color = color.copy(alpha = 0.25f), radius = 16f, center = center)
                drawCircle(color = color, radius = 6f, center = center)
                drawCircle(
                    color = Color.White.copy(alpha = 0.7f),
                    radius = 6f, center = center,
                    style = Stroke(width = 1.5f)
                )

                if (isDebugMode) {
                    drawCircle(
                        color = color.copy(alpha = 0.5f),
                        radius = 11f, center = center,
                        style = Stroke(width = 1.5f)
                    )
                }
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

        val ballColor = Color(0xFFFF6D00)
        val coreAlpha = ballPosition.confidence.coerceIn(0.3f, 1f)
        val glowAlpha = (ballPosition.confidence * 0.4f).coerceIn(0.05f, 0.4f)

        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(ballColor.copy(alpha = glowAlpha), ballColor.copy(alpha = 0f)),
                center = center, radius = 28f
            ),
            radius = 28f, center = center
        )

        drawCircle(color = ballColor.copy(alpha = coreAlpha), radius = 7f, center = center)
        drawCircle(
            color = Color.White.copy(alpha = coreAlpha * 0.8f),
            radius = 7f, center = center,
            style = Stroke(width = 1.8f)
        )

        val crossLen = 14f
        val crossColor = ballColor.copy(alpha = coreAlpha * 0.6f)
        drawLine(crossColor, Offset(center.x - crossLen, center.y), Offset(center.x - 9f, center.y), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x + 9f, center.y), Offset(center.x + crossLen, center.y), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x, center.y - crossLen), Offset(center.x, center.y - 9f), strokeWidth = 1.2f, cap = StrokeCap.Round)
        drawLine(crossColor, Offset(center.x, center.y + 9f), Offset(center.x, center.y + crossLen), strokeWidth = 1.2f, cap = StrokeCap.Round)

        if (isDebugMode) {
            val confRadius = 12f + ballPosition.confidence * 8f
            drawCircle(
                color = ballColor.copy(alpha = 0.35f),
                radius = confRadius, center = center,
                style = Stroke(width = 1.5f)
            )
        }
    }
}

// --- Camera Moved Alert ---
@Composable
private fun CameraMovedAlert(onDismiss: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .background(ShotColors.alertOrange.copy(alpha = 0.15f), RoundedCornerShape(16.dp))
            .border(1.dp, ShotColors.alertOrange.copy(alpha = 0.4f), RoundedCornerShape(16.dp))
            .padding(horizontal = 28.dp, vertical = 20.dp)
    ) {
        Text("!", color = ShotColors.alertOrange, fontSize = 28.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Camera Moved", color = ShotColors.textPrimary, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            "Court position may have shifted.\nTap to re-detect.",
            color = ShotColors.textSecondary, fontSize = 12.sp, textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "RE-DETECT",
            color = ShotColors.alertOrange,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp,
            modifier = Modifier
                .background(ShotColors.alertOrange.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
                .border(1.dp, ShotColors.alertOrange.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) { onDismiss() }
                .padding(horizontal = 20.dp, vertical = 8.dp)
        )
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
        Canvas(modifier = Modifier.size(48.dp)) {
            val w = size.width; val h = size.height
            val lineColor = ShotColors.textSecondary; val stroke = 1.5f
            drawRect(color = lineColor, style = Stroke(stroke))
            drawLine(lineColor, Offset(0f, h / 2), Offset(w, h / 2), stroke)
            drawLine(lineColor, Offset(w * 0.2f, h * 0.25f), Offset(w * 0.8f, h * 0.25f), stroke)
            drawLine(lineColor, Offset(w * 0.2f, h * 0.75f), Offset(w * 0.8f, h * 0.75f), stroke)
            drawLine(lineColor, Offset(w / 2, h * 0.25f), Offset(w / 2, h * 0.75f), stroke)
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            stringResource(R.string.camera_guide_center),
            color = ShotColors.textPrimary, fontSize = 15.sp,
            fontWeight = FontWeight.Medium, letterSpacing = 0.2.sp
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            "Position your camera behind the baseline",
            color = ShotColors.textSecondary, fontSize = 12.sp, letterSpacing = 0.1.sp
        )
    }
}

// --- Debug Panel ---
@Composable
private fun DebugPanel(
    result: com.shot.core.model.CourtDetectionResult,
    fps: Float,
    ballPosition: BallTrackingDetector.BallPosition?,
    isCourtLocked: Boolean,
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
        if (isCourtLocked) {
            DebugRow("Court", "LOCKED", ShotColors.lockBlue)
        }
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
private fun DebugRow(label: String, value: String, valueColor: Color = ShotColors.textMono) {
    Row(
        modifier = Modifier.padding(vertical = 1.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = ShotColors.textSecondary, fontSize = 10.sp,
            fontFamily = FontFamily.Monospace, modifier = Modifier.width(72.dp))
        Text(value, color = valueColor, fontSize = 10.sp,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Medium)
    }
}

// --- Bottom Bar ---
@Composable
private fun BottomBar(
    isDebugMode: Boolean,
    isCourtLocked: Boolean,
    canLock: Boolean,
    onToggleDebug: () -> Unit,
    onToggleLock: () -> Unit,
    onCapture: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
    ) {
        // Debug toggle
        BottomChip(
            text = "DEBUG",
            isActive = isDebugMode,
            activeColor = ShotColors.accentTeal,
            onClick = onToggleDebug
        )

        // Court Lock toggle
        val lockColor = when {
            isCourtLocked -> ShotColors.lockBlue
            canLock -> ShotColors.textSecondary
            else -> ShotColors.textSecondary.copy(alpha = 0.4f)
        }
        BottomChip(
            text = if (isCourtLocked) "UNLOCK" else "LOCK",
            isActive = isCourtLocked,
            activeColor = ShotColors.lockBlue,
            inactiveColor = lockColor,
            enabled = canLock || isCourtLocked,
            onClick = onToggleLock
        )

        // Screenshot capture button (circle)
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(40.dp)
                .background(ShotColors.surfaceGlass, CircleShape)
                .border(2.dp, ShotColors.captureWhite.copy(alpha = 0.6f), CircleShape)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) { onCapture() }
        ) {
            // Inner filled circle (camera shutter style)
            Box(
                modifier = Modifier
                    .size(30.dp)
                    .background(ShotColors.captureWhite.copy(alpha = 0.85f), CircleShape)
            )
        }
    }
}

@Composable
private fun BottomChip(
    text: String,
    isActive: Boolean,
    activeColor: Color,
    inactiveColor: Color = ShotColors.textSecondary,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Text(
        text = text,
        color = if (isActive) activeColor else inactiveColor,
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold,
        fontFamily = FontFamily.Monospace,
        letterSpacing = 1.sp,
        modifier = Modifier
            .background(
                if (isActive) activeColor.copy(alpha = 0.12f) else ShotColors.surfaceGlass,
                RoundedCornerShape(8.dp)
            )
            .border(
                1.dp,
                if (isActive) activeColor.copy(alpha = 0.3f) else ShotColors.glassBorder,
                RoundedCornerShape(8.dp)
            )
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                enabled = enabled
            ) { onClick() }
            .padding(horizontal = 16.dp, vertical = 8.dp)
    )
}
