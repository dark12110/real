package com.example.a3app.data.camera

import android.content.Context
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * Camera manager for the A3 app
 */
class CameraManager(
    private val context: Context,
    private val executor: Executor = ContextCompat.getMainExecutor(context)
) {
    private var imageCapture: ImageCapture? = null
    
    /**
     * Start camera preview
     */
    suspend fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: androidx.camera.view.PreviewView,
        cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    ) = withContext(Dispatchers.Main) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = suspendCoroutine<ProcessCameraProvider> { continuation ->
            cameraProviderFuture.addListener({
                try {
                    val provider = cameraProviderFuture.get()
                    continuation.resume(provider)
                } catch (e: Exception) {
                    continuation.resumeWithException(e)
                }
            }, executor)
        }
        
        // Preview use case
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        
        // Image capture use case
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        
        try {
            // Unbind all use cases before rebinding
            cameraProvider.unbindAll()
            
            // Bind use cases to camera
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageCapture
            )
        } catch (e: Exception) {
            throw CameraException("Failed to start camera", e)
        }
    }
    
    /**
     * Take photo and save to file
     */
    suspend fun takePhoto(outputDirectory: File): Uri = suspendCoroutine { continuation ->
        val imageCapture = imageCapture ?: throw CameraException("Camera not initialized")
        
        // Create output file
        val photoFile = createFile(outputDirectory, "jpg")
        
        // Create output options object
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        
        // Set up image capture listener
        imageCapture.takePicture(
            outputOptions,
            executor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val savedUri = outputFileResults.savedUri ?: Uri.fromFile(photoFile)
                    continuation.resume(savedUri)
                }
                
                override fun onError(exception: ImageCaptureException) {
                    continuation.resumeWithException(
                        CameraException("Failed to capture image", exception)
                    )
                }
            }
        )
    }
    
    /**
     * Create file with timestamp
     */
    private fun createFile(baseFolder: File, extension: String): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        return File(baseFolder, "IMG_${timestamp}.$extension")
    }
    
    /**
     * Get output directory for photos
     */
    fun getOutputDirectory(): File {
        val mediaDir = context.externalMediaDirs.firstOrNull()?.let {
            File(it, "A3App").apply { mkdirs() }
        }
        return if (mediaDir != null && mediaDir.exists()) mediaDir else context.filesDir
    }
    
    /**
     * Switch between front and back camera
     */
    fun toggleCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: androidx.camera.view.PreviewView,
        currentCameraSelector: CameraSelector
    ) {
        val newCameraSelector = if (currentCameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        
        // Restart camera with new selector
        kotlinx.coroutines.runBlocking {
            startCamera(lifecycleOwner, previewView, newCameraSelector)
        }
    }
}

/**
 * Custom exception for camera operations
 */
class CameraException(message: String, cause: Throwable? = null) : Exception(message, cause)
