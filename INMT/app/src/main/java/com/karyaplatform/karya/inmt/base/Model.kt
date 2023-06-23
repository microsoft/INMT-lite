package com.karyaplatform.karya.inmt.base

abstract class Model() {
    abstract fun run(
        sourceSentence: String,
        partialSentence: String,
        forward: Int,
        depth: Int
    ): Array<String>
}