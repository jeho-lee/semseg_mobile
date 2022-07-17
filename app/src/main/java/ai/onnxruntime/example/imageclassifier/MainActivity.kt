// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.lang.Runnable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    private val labelData: List<String> by lazy { readLabels() }
    private var labelMapColor: MutableList<Int> = generateRandomColors()

    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    private var enableQuantizedModel: Boolean = false

    /** Model Configs Setting */
    //        return resources.openRawResource(R.raw.pascal_context_labels_homepage).bufferedReader().readLines()
    //        return resources.openRawResource(R.raw.cityscapes_labels).bufferedReader().readLines()
    val fClassLabel: Int = R.raw.cityscapes_labels

    //        val modelID = if (enableQuantizedModel) R.raw.mobilenet_v2_uint8 else R.raw.mobilenet_v2_float
    //        val modelID = R.raw.deeplabv3plus_onnxruntime
    val fModelCheckpoint: Int = R.raw.pspnet_cityscapes_onnxruntime

    val inputWidth: Int = 512
    val inputHeight: Int = 1024

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request Camera permission
        if (allPermissionsGranted()) {
            ortEnv = OrtEnvironment.getEnvironment()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

//        enable_quantizedmodel_toggle.setOnCheckedChangeListener { _, isChecked ->
//            enableQuantizedModel = isChecked
//            setORTAnalyzer()
//        }


    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            setORTAnalyzer()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }

        }
    }

    private fun updateUI(result: Result) {
//        if (result.detectedScore.isEmpty())
//            return

        if (result.predictedPixels.isEmpty()) {
            return
        }

        runOnUiThread {
            /*
            percentMeter.progress = (result.detectedScore[0] * 100).toInt()
            detected_item_1.text = labelData[result.detectedIndices[0]]
            detected_item_value_1.text = "%.2f%%".format(result.detectedScore[0] * 100)

            if (result.detectedIndices.size > 1) {
                detected_item_2.text = labelData[result.detectedIndices[1]]
                detected_item_value_2.text = "%.2f%%".format(result.detectedScore[1] * 100)
            }

            if (result.detectedIndices.size > 2) {
                detected_item_3.text = labelData[result.detectedIndices[2]]
                detected_item_value_3.text = "%.2f%%".format(result.detectedScore[2] * 100)
            }
            */

            // Init
            predictedLabels.removeAllViews()

            val detectedIndicesMap: MutableMap<Int, String> = mutableMapOf()

            val resultBitmap: Bitmap = Bitmap.createBitmap(inputHeight, inputWidth, Bitmap.Config.ARGB_8888)
            for (x in 0 until inputHeight) {
//                println()
                for (y in 0 until inputWidth) {
//                    print(result.predictedPixels[x][y].toInt().toString() + " ")

                    // Push detected labels from current frame
                    val detectedIndex: Int = result.predictedPixels[y][x].toInt()
                    detectedIndicesMap[detectedIndex] = labelData[detectedIndex]

                    // Draw segmentation result
                    val labelColor = labelMapColor[detectedIndex]
                    resultBitmap.setPixel(x, y, labelColor)
                }
            }

            /** Update detected labels from current frame */
            for ((labelIndex, labelName) in detectedIndicesMap) {
                val labelInfoContainer = LinearLayout(this)
                labelInfoContainer.orientation = LinearLayout.HORIZONTAL

                val rectView = Button(this)
                rectView.width = 10
                rectView.height = 6
                rectView.setBackgroundColor(labelMapColor[labelIndex])

                val textView = TextView(this)
                textView.text = labelName

                labelInfoContainer.addView(rectView)
                labelInfoContainer.addView(textView)

                predictedLabels.addView(labelInfoContainer)
            }

//            for (x in 0 until 512) {
//                println()
//                for (y in 0 until 512) {
//                    print(resultBitmap.getPixel(x, y).toString() + " ")
//                }
//            }

//            for (i in 0 until 59) {
//                println("cur color val: " + labelMapColor.indexOf(i).toString())
//            }

            predictionResultImage.setImageBitmap(resultBitmap)

            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }


    }

    private fun generateRandomColors(): MutableList<Int> {
        val colors: MutableList<Int> = mutableListOf()

        colors.add(Color.argb(255, 0, 0, 0)) // background color is black

        for (i in 0 until 59) {
            val r: Int = (0..256).random()
            val g: Int = (0..256).random()
            val b: Int = (0..256).random()

//            println("cur color val: " + Color.argb(255, r, g, b).toString())

            colors.add(Color.argb(255, r, g, b))
        }

        for (i in 0 until 59) {
            println("cur color val: " + colors[i].toString())
        }

        return colors
    }

    // Read pascal context labels
    private fun readLabels(): List<String> {
        return resources.openRawResource(fClassLabel).bufferedReader().readLines()
    }

    // Read ort model into a ByteArray, run in background
    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        val modelID = fModelCheckpoint

        resources.openRawResource(modelID).readBytes()
    }

    // Create a new ORT session in background
    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        ortEnv?.createSession(readModel())
    }

    // Create a new ORT session and then change the ImageAnalysis.Analyzer
    // This part is done in background to avoid blocking the UI
    private fun setORTAnalyzer() {
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            imageAnalysis?.setAnalyzer(
                backgroundExecutor,
                ORTAnalyzer(createOrtSession(), ::updateUI)
            )
        }
    }

    companion object {
        const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
