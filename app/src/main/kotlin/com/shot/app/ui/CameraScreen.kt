package com.shot.app.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
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

@Composable
fun CameraScreen(
    viewModel: CameraViewModel = hiltViewModel()
) {
    val detectionResult by viewModel.detectionResult.collectAsState()
    val isDebugMode by viewModel.isDebugMode.collectAsState()

    // FPS calculation
    var fps by remember { mutableFloatStateOf(0f) }
    var lastFrameTime by remember { mutableLongStateOf(0L) }
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

        // Layer 2: Court Overlay
        CourtOverlay(
            projectedKeypoints = detectionResult.projectedKeypoints,
            detectedKeypoints = detectionResult.detectedKeypoints,
            isDebugMode = isDebugMode
        )

        // Layer 3: Status Indicator (tap to toggle debug)
        StatusIndicator(
            status = detectionResult.status,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
                .clickable { viewModel.toggleDebugMode() }
        )

        // Layer 4: Debug Info
        if (isDebugMode) {
            DebugOverlay(
                result = detectionResult,
                fps = fps,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
            )
        }

        // Layer 5: Camera Guide (shown when not detected)
        if (detectionResult.status == DetectionStatus.NOT_DETECTED) {
            Text(
                text = stringResource(R.string.camera_guide_center),
                color = Color.White,
                fontSize = 18.sp,
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(32.dp)
            )
        }

        // Layer 6: Bottom controls
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
        ) {
            // Debug toggle button
            Text(
                text = if (isDebugMode) "DEBUG ON" else "DEBUG",
                color = if (isDebugMode) Color.Green else Color.Gray,
                fontSize = 12.sp,
                modifier = Modifier
                    .background(
                        Color.Black.copy(alpha = 0.5f),
                        RoundedCornerShape(4.dp)
                    )
                    .clickable { viewModel.toggleDebugMode() }
                    .padding(horizontal = 12.dp, vertical = 6.dp)
            )
        }
    }
}

@Composable
private fun CourtOverlay(
    projectedKeypoints: List<Keypoint>,
    detectedKeypoints: List<Keypoint>,
    isDebugMode: Boolean
) {
    if (projectedKeypoints.isEmpty()) return

    val keypointMap = projectedKeypoints.associateBy { it.id }

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

        // Draw court lines
        val lineColor = Color.Green.copy(alpha = 0.8f)
        val lineWidth = 3f

        for ((startId, endId) in ItfCourtSpec.COURT_LINES) {
            val start = keypointMap[startId] ?: continue
            val end = keypointMap[endId] ?: continue
            drawLine(
                color = lineColor,
                start = Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
                end = Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
                strokeWidth = lineWidth
            )
        }

        // Draw keypoint dots with confidence colors
        for (kp in detectedKeypoints) {
            val color = when (ConfidenceLevel.from(kp.confidence)) {
                ConfidenceLevel.HIGH -> Color.Green
                ConfidenceLevel.MEDIUM -> Color.Yellow
                ConfidenceLevel.LOW -> Color.Red
            }
            drawCircle(
                color = color,
                radius = 8f,
                center = Offset(kp.x * scaleX + offsetX, kp.y * scaleY + offsetY)
            )
            if (isDebugMode) {
                drawCircle(
                    color = color,
                    radius = 12f,
                    center = Offset(kp.x * scaleX + offsetX, kp.y * scaleY + offsetY),
                    style = Stroke(width = 2f)
                )
            }
        }
    }
}

@Composable
private fun StatusIndicator(
    status: DetectionStatus,
    modifier: Modifier = Modifier
) {
    val (text, color) = when (status) {
        DetectionStatus.DETECTED -> stringResource(R.string.court_detected) to Color.Green
        DetectionStatus.PARTIAL -> stringResource(R.string.court_partial) to Color.Yellow
        DetectionStatus.NOT_DETECTED -> stringResource(R.string.court_not_detected) to Color.Red
    }

    Text(
        text = text,
        color = color,
        fontSize = 14.sp,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    )
}

@Composable
private fun DebugOverlay(
    result: com.shot.core.model.CourtDetectionResult,
    fps: Float,
    modifier: Modifier = Modifier
) {
    val debugText = buildString {
        appendLine("FPS: ${"%.1f".format(fps)}")
        appendLine("Inference: ${result.inferenceTimeMs} ms")
        appendLine("Reproj: ${"%.1f".format(result.reprojectionError)} px")
        appendLine("KP: ${result.reliableKeypointCount}/8 | Proj: ${result.projectedKeypoints.size}/16")
        appendLine("Status: ${result.status.name}")
    }

    Text(
        text = debugText,
        color = Color.White,
        fontSize = 11.sp,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
            .padding(8.dp)
    )
}
