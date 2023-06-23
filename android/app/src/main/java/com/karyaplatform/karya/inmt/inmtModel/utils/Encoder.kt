package com.karyaplatform.sentencepiece.inmtModel.utils

import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList

class Encoder(encoderFileDescriptor: AssetFileDescriptor) {
    private val ENCODER_INPUT_SIZE = 28
    private val ENCODER_OUTPUT_SIZE = 512

    private val modelEncoderInterpreter: Interpreter

    init {
        val loadedModelFile = Common.loadModelFile(encoderFileDescriptor)
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            // if the GPU is not supported, run on 4 threads
//            this.setNumThreads(4)
        }
        try {
//            modelEncoderInterpreter = Interpreter(loadedModelFile!!, options)
            modelEncoderInterpreter = Interpreter(loadedModelFile!!, options)
        } catch (e: Exception) {
            Log.e("INMT_ENCODER_ERROR", e.toString())
            throw Error("FAILED TO INITIALIZE ENCODER")
        }
    }

    /**
     * Function to run encoder for input
     */
    fun run(ids: MutableList<Int>): Array<Array<FloatArray>> {

        Log.i("MODEL ENCODER: ", "STARTED")
        // Set the shapes of output and input arrays
        val outputVal: HashMap<Int, Any> = HashMap()
        outputVal[0] = Array(1) { Array(ENCODER_INPUT_SIZE) { FloatArray(ENCODER_OUTPUT_SIZE) } }
        val attention = Array(1) {
            IntArray(
                28
            )
        }
        val tokenInput = Array(1) {
            IntArray(
                28
            )
        }
        // Initialising the input arrays
        for (idx in ids.indices) {
            tokenInput[0][idx] = ids[idx]
            attention[0][idx] = 1
        }
        // Run the model
        modelEncoderInterpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(
                tokenInput,
                attention
            ), outputVal
        )
        Log.i("MODEL ENCODER: ", "RUN SUCCESS")

        //Return the result
        return outputVal[0] as Array<Array<FloatArray>>
    }
}