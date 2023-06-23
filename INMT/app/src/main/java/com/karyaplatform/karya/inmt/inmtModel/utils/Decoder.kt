package com.karyaplatform.sentencepiece.inmtModel.utils

import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList

class Decoder(decoderFileDescriptor: AssetFileDescriptor) {

    private val DECODER_OUTPUT_ARRAY_SIZE = 28
    private val DECODER_ENBEDDING_SIZE = 38835

    private val modelDecoderInterpreter: Interpreter

    init {
        // Initialize the decoder
        val loadedModelFile = Common.loadModelFile(decoderFileDescriptor)
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            // if the GPU is not supported, run on 4 threads
//            this.setNumThreads(4)
        }

        try {
            modelDecoderInterpreter = Interpreter(loadedModelFile!!, options)
        } catch (e: java.lang.Exception) {
            Log.e("MODEL_DECODER", e.toString())
            throw Error("FAILED TO INITIALISE DECODER")
        }
    }

    fun run(encoderResult: Array<Array<FloatArray>>, input: Array<IntArray>, itr: Int): FloatArray {
        Log.i("MODEL DECODER: ", "STARTED")
        // Set the shapes of output and input arrays
        val outputVal: HashMap<Int, Any> = HashMap()
        // TODO: CHECK THIS
        outputVal[0] =
            Array(1) { Array(DECODER_OUTPUT_ARRAY_SIZE) { FloatArray(DECODER_ENBEDDING_SIZE) } }
        // Run the model
        val t1 = System.currentTimeMillis()
        modelDecoderInterpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(
                input,
                encoderResult
            ), outputVal
        )
        val t2 = System.currentTimeMillis()
        Log.i("MODEL_DECODER: ", "RUN SUCCESS: ${t2 - t1}")
        modelDecoderInterpreter.resetVariableTensors()
        //Return the result
        return (outputVal[0] as Array<Array<FloatArray>>)[0][itr]
    }
}