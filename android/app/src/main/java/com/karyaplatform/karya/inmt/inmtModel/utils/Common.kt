package com.karyaplatform.sentencepiece.inmtModel.utils

import android.content.res.AssetFileDescriptor
import android.util.Log
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Common {
    companion object {
        internal fun loadModelFile(fileDescriptor: AssetFileDescriptor): MappedByteBuffer? {
            Log.i("Model read:", "started")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            Log.i("Model read:", "success")
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
}