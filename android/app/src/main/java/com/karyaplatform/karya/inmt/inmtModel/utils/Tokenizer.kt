package com.karyaplatform.sentencepiece.inmtModel.utils

import com.github.google.sentencepiece.SentencePieceProcessor
import java.io.InputStream

class Tokenizer {
    companion object {
        fun initialiseTokenizer(
            modelStream: InputStream,
            sentencePieceProcessor: SentencePieceProcessor
        ): SentencePieceProcessor {
            val content: ByteArray = modelStream.readBytes()
            sentencePieceProcessor.loadFromSerializedProto(content)
            return sentencePieceProcessor
        }

        fun SentencePieceProcessor.loadFromFileInputStream(modelStream: InputStream) {
            val content: ByteArray = modelStream.readBytes()
            this.loadFromSerializedProto(content)
        }

        /**
         * Return ids for a sentence according to the provided subWord Encoder
         */
        fun SentencePieceProcessor.encodeWithSubWordEncoder(subWordEncoder: HashMap<String, Int>, sentence: String): MutableList<Int> {
            val ids = mutableListOf<Int>()
            this.encodeAsPieces(sentence).forEach { piece -> ids.add(subWordEncoder[piece]!!) }
            return ids
        }

        /**
         * Return subWords for a sentence according to the provided subWord decoder
         */
        fun SentencePieceProcessor.decodeWithSubWordDecoder(subWordDecoder: HashMap<Int, String>, ids: IntArray): String? {
            val tokens = mutableListOf<String>()
            ids.forEach { tokenId -> tokens.add(subWordDecoder[tokenId]!!) }
            return this.decodePieces(tokens)
        }

        fun SentencePieceProcessor.decodeIds(vararg ids: Int): String? {
            val mappedIds = ids.map {
                2// TODO: MAP IDS
            }
            return this.decodeIds(*mappedIds.toIntArray())
        }
    }
}