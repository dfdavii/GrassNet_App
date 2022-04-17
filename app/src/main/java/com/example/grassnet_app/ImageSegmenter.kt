package com.example.grassnet_app

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Image segmenter
 *
 * @author sercant
 * @date 05/12/2018
 */
class ImageSegmenter(
    private val activity: Activity
) {

    companion object {
        const val TAG: String = "ImageSegmenter"
        /** Dimensions of inputs.  */
        const val DIM_BATCH_SIZE = 1

        const val DIM_PIXEL_SIZE = 3
    }

    val currentModel: Model
        get() = modelList[currentModelIndex]

    /** Pre-allocated buffers for storing image data in.  */
    private var intValues = IntArray(0)
    private var outFrame = IntArray(0)

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null
    private val modelList = arrayOf(
        Model(
            "grass1", // grassNet_mobilenet_v2
            Model.Dataset.PASCAL,
            256, 256, 256, 256
        ),
        Model(
            "grassNet_mobilenet_v2", // grassNet_mobilenet_v2
            Model.Dataset.PASCAL,
            256, 256, 256, 256
        )
    )

    private var currentModelIndex = 0

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter? = null

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    private lateinit var imgData: ByteBuffer

    private lateinit var segmentedImage: ByteBuffer

    init {
        loadModel(0)
        Log.d(TAG, "Created a Tensorflow Lite Image Segmenter.")
    }

    fun changeModel() {
        close()
        currentModelIndex = ++currentModelIndex % modelList.size
        loadModel(currentModelIndex)
    }

    private fun loadModel(index: Int) {
        if (index >= 0 && index < modelList.size) {
            currentModelIndex = index
            tfliteModel = loadModelFile(activity)
            recreateInterpreter()
        }
    }

    private fun recreateInterpreter() {
        val model = modelList[currentModelIndex]
        tflite?.close()
        tfliteModel?.let {
            tflite = Interpreter(it, tfliteOptions)
        }
        imgData = ByteBuffer
            .allocateDirect(DIM_BATCH_SIZE * model.inputWidth * model.inputHeight * DIM_PIXEL_SIZE * getNumBytesPerChannel())
            .also {
                it.order(ByteOrder.nativeOrder())
            }

        segmentedImage = ByteBuffer
            .allocateDirect(DIM_BATCH_SIZE * model.outputWidth * model.outputHeight * getNumBytesPerChannel())
            .also {
                it.order(ByteOrder.nativeOrder())
            }

        outFrame = IntArray(model.outputWidth * model.outputHeight)
        intValues = IntArray(model.inputWidth * model.inputHeight)
    }

    // fun setUseNNAPI(nnapi: Boolean) {
    //     tfliteOptions.setUseNNAPI(nnapi)
    //     recreateInterpreter()
    // }

    fun setNumThreads(numThreads: Int) {
        tfliteOptions.setNumThreads(numThreads)
        recreateInterpreter()
    }

    /** Closes tflite to release resources.  */
    fun close() {
        tflite?.close()
        tflite = null
        tfliteModel?.clear()
        tfliteModel = null
    }

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(getModelPath())
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData.rewind()

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        for (pixel in 0 until intValues.size) {
            val value = intValues[pixel]

            imgData.apply {
                putFloat((value shr 16 and 0xff).toFloat())
                putFloat((value shr 8 and 0xff).toFloat())
                putFloat((value and 0xff).toFloat())
            }
        }
    }

    /**
     * Get the name of the model file stored in Assets.
     *
     * @return
     */
    private fun getModelPath(): String = "${modelList[currentModelIndex].path}.tflite"

    /**
     * Get the number of bytes that is used to store a single color channel value.
     *
     * @return
     */
    private fun getNumBytesPerChannel(): Int = 4

    /**
     * Segments a frame from the preview stream.
     */
    fun segmentFrame(bitmap: Bitmap): IntArray {
        if (tflite == null) {
            Log.e(TAG, "Image segmenter has not been initialized; Skipped.")
        }

        convertBitmapToByteBuffer(bitmap)

        segmentedImage.rewind()
        tflite?.run(imgData, segmentedImage)

        segmentedImage.position(0)
        var i = 0
        while (segmentedImage.hasRemaining())
            outFrame[i++] = segmentedImage.int

        return outFrame
    }

    data class Model(
        val path: String,
        val colorSchema: Dataset,
        val inputWidth: Int,
        val inputHeight: Int,
        val outputWidth: Int,
        val outputHeight: Int
    ) {
        val colors: Array<Int> = when (colorSchema) {
            Dataset.PASCAL -> arrayOf(
                0x00FFFFFF.toInt(),
                0xFF228B22.toInt(), // green
                0xFFFF1493.toInt(), // pink

            )
        }

        enum class Dataset {
            PASCAL
        }
    }
}