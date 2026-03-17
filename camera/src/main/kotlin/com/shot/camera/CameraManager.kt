package com.shot.camera

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages CameraX setup for court detection.
 *
 * Configures:
 * - Preview: displayed in PreviewView
 * - ImageAnalysis: KEEP_ONLY_LATEST backpressure (auto frame drop)
 * - Back camera, standard lens (not ultra-wide)
 */
class CameraManager(
    private val context: Context,
) {

    private var cameraProvider: ProcessCameraProvider? = null
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    /**
     * Bind camera to lifecycle with preview and analysis use cases.
     *
     * @param previewView The PreviewView to display camera feed
     * @param lifecycleOwner Activity/Fragment lifecycle
     * @param onFrame Callback for each analyzed frame
     */
    fun bind(
        previewView: PreviewView,
        lifecycleOwner: LifecycleOwner,
        onFrame: (ImageProxy) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            // Preview use case
            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            // ImageAnalysis use case
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        onFrame(imageProxy)
                        // Caller must close imageProxy when done
                    }
                }

            // Camera selector: back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Bind use cases
            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(context))
    }

    fun unbind() {
        cameraProvider?.unbindAll()
    }

    fun shutdown() {
        unbind()
        analysisExecutor.shutdown()
    }
}
