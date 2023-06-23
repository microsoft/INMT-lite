package com.karyaplatform.karya.inmt.inmtModel

import android.content.res.AssetFileDescriptor
import android.util.Log
import com.karyaplatform.karya.inmt.base.Model
import com.karyaplatform.sentencepiece.inmtModel.utils.Decoder
import com.karyaplatform.sentencepiece.inmtModel.utils.Encoder
import com.karyaplatform.sentencepiece.inmtModel.utils.Tokenizer.Companion.decodeWithSubWordDecoder
import com.karyaplatform.sentencepiece.inmtModel.utils.Tokenizer.Companion.encodeWithSubWordEncoder
import com.karyaplatform.sentencepiece.inmtModel.utils.Tokenizer.Companion.loadFromFileInputStream
import com.github.google.sentencepiece.SentencePieceProcessor
import org.json.JSONObject
import java.io.InputStream
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.math.min

class NgINMTModel(
    vocabStream: InputStream,
    sourceSpieceModelFileInputStream: InputStream,
    targetSpieceModelFileInputStream: InputStream,
    encoderModelFileDescriptor: AssetFileDescriptor,
    decoderModelFileDescriptor: AssetFileDescriptor,
) : Model() {
    private val subWordEncoder: HashMap<String, Int> = HashMap()
    private val subWordDecoder: HashMap<Int, String> = HashMap()
    private val sourceSpieceProcessor: SentencePieceProcessor
    private val targetSpieceProcessor: SentencePieceProcessor
    private val encoder: Encoder
    private val decoder: Decoder
    // TODO: Change this to space
    // Ascii of the word separator in sentencepiece.
    private val wordSeparatorAscii = 9601
    private val START_TOKEN = 1
    private val END_OF_SENTENCE = 2

    init {
        val vocabJson = JSONObject(vocabStream.bufferedReader().use { it.readText() })
        // Iterate over the object to create subWord encoder and decoder hashmaps
        for (token in vocabJson.keys()) {
            val tokenId = vocabJson.getInt(token)
            subWordEncoder[token]= tokenId
            subWordDecoder[tokenId] = token
        }
        sourceSpieceProcessor = SentencePieceProcessor()
        targetSpieceProcessor = SentencePieceProcessor()
        sourceSpieceProcessor.loadFromFileInputStream(sourceSpieceModelFileInputStream)
        targetSpieceProcessor.loadFromFileInputStream(targetSpieceModelFileInputStream)
        encoder = Encoder(encoderModelFileDescriptor)
        decoder = Decoder(decoderModelFileDescriptor)
        Log.i("PIECE TEST", sourceSpieceProcessor.decodePieces(mutableListOf("pol", "_lol")))
        Log.i("VOCAB_SIZE", targetSpieceProcessor.pieceSize.toString())
    }

    override fun run(sourceSentence: String, partialSentence: String, forward: Int, depth: Int): Array<String> {
        /**
         * We are only covering following cases:
         * 1. Depth is 1 and any forward value
         * 2. Forward is 1 and any depth value
         */
        // TODO: Generalise this function for any depth and any forward value


        var modelOutput =
            Array(depth) { Triple(Array(1) { IntArray(28) }, 1F, -1) }

        // Tokenize the String
        val encodedSource = sourceSpieceProcessor.encodeAsIds(sourceSentence).toMutableList()
        encodedSource.add(END_OF_SENTENCE)
        // Run the encoder
        val encoderResult = encoder.run(encodedSource)
        // Tokenize partial sentence
        val encodedPartial = targetSpieceProcessor.encodeWithSubWordEncoder(subWordEncoder, partialSentence)

        val firstDecoderInput = Array(1) { IntArray(28) }
        // Set the start token
        firstDecoderInput[0][0] = START_TOKEN
        // Set the first decoder input from the partial input
        val iterateUpto = min(firstDecoderInput[0].size-1, encodedPartial.size)
        for (idx in 0 until iterateUpto) {
            firstDecoderInput[0][idx+1] = encodedPartial[idx]
        }
        val itr_start = encodedPartial.size
        // Run the decoder for first token
        // ITR = 0 because this is the first iteration of decoding
        val decoderResult = decoder.run(encoderResult, firstDecoderInput, itr_start)
        val topSubs = findTopNIndex(decoderResult, depth)

        for(idx in topSubs.indices) {
            // ITR = 1 because decoder has been invoked the first time in line 59
            modelOutput[idx] = Triple(createDecoderInputWithNewToken(firstDecoderInput, topSubs[idx].first, itr_start), topSubs[idx].second, itr_start+1)
        }

        // Cap steps to 28
        val steps = min(forward, 28)

        for (step in 1 until steps) {
            val heap = getBeamSearchHeap(depth*depth)
            for (d in 0 until depth) {
                val nextWords = getNextWords(encoderResult, modelOutput[d].first, modelOutput[d].second, depth, modelOutput[d].third)
                for (eachWord in nextWords) {
                    heap.add(Triple(eachWord.second, eachWord.first, eachWord.third))
                }
            }
            val topSequences = ArrayList<Triple<Float, Array<IntArray>, Int>>()
            for (i in 0 until depth) topSequences.add(heap.poll())

            val tempModelOutput = Array(depth) { Triple(Array(1) { IntArray(28) }, 1F, -1) }
            for (idx in topSequences.indices) {
                tempModelOutput[idx] = Triple(topSequences[idx].second, topSequences[idx].first, topSequences[idx].third)
                modelOutput = tempModelOutput
            }
        }

        val result = Array<String>(depth) { "" }

        for (idx in result.indices) {
            result[idx] = targetSpieceProcessor.decodeWithSubWordDecoder(subWordDecoder, modelOutput[idx].first[0])!!
        }

        return result
    }

    /**
     * Create decoder input by adding new token at [itr+1] position of the provided decoder input
     * @param decoderInput: Decoder Input to which the new token will be added
     * @param token: The token to be added to the decoderInput
     * @param itr: Number of times decoder has been invoked (or would have been invoked) to generate the decoderInput
     */
    private fun createDecoderInputWithNewToken(decoderInput: Array<IntArray>, token: Int, itr: Int): Array<IntArray> {
        val copy = Array(1) { IntArray(28) }
        for (idx in copy[0].indices) {
            copy[0][idx] = decoderInput[0][idx]
        }
        copy[0][itr+1] = token
        return copy
    }

    fun findTopNIndex(array: FloatArray, n: Int): Array<Pair<Int, Float>> {
        // Here the pair's first member represents the index and second represents the value of the array at the index
        val comparator = Comparator<Pair<Int, Float>> { t1, t2 ->
            if (t2.second > t1.second) 1
            else if (t1.second > t2.second) - 1
            else 0
        }
        val pq = PriorityQueue(array.size, comparator)
        for (i in array.indices) pq.add(Pair(i, array[i]))

        val result = Array(n) { Pair(-1, -9999999F) }
        // Pop the top n elements from heap
        for (i in 0 until n) result[i] = pq.poll()

        return result
    }

    fun getBeamSearchHeap(size: Int): PriorityQueue<Triple<Float, Array<IntArray>, Int>> {
        // Here the pair's:
        // 1. First member represents the word probability
        // 2. Second represents the array of indexes (subwords) in form of decoderInput that represent the word
        // 3. Third represents the index of the words so far in modelOutput Array
        val comparator = Comparator<Triple<Float, Array<IntArray>, Int>> { t1, t2 ->
            if (t2.first > t1.first) 1
            else if (t1.first > t2.first) -1
            else 0
        }
        return PriorityQueue(size, comparator)
    }

    /**
     * The function predicts next N most likely words given sequence of sub tokens
     * @param encoderResult: encoder output for source sequence
     * @param decoderInput: The decoder's input for which to predict next words
     * @param prob: Probability for the [decoderInput] passed to the function
     * @param number: Number of next most probable words by the model
     * @param itr: The number of times the decoder has been invoked to obtain [decoderInput]
     */
    private fun getNextWords(
        encoderResult: Array<Array<FloatArray>>,
        decoderInput: Array<IntArray>,
        prob: Float,
        number: Int,
        itr: Int
    ): ArrayList<Triple<Array<IntArray>, Float, Int>> {
        // Description of the member of words Arraylist:
        // Each member (called word) is Triple consisting of following:
        // 1. first (decoderInput): Represents the decoderInput combined with the decoderInput provided to the function's argument to construct that word.
        // 2. second (probability): Represents the resultant probability of the sequence upto (including) the generated word.
        // 3. third (itr (iteration)): Represents the number of times the decoder was invoked to form the sequence.
        val words = ArrayList<Triple<Array<IntArray>, Float, Int>>()
        // First model invocation
        val decoderResult = decoder.run(encoderResult, decoderInput, itr)
        // Get top n sub words with their probability
        val topSubWords = findTopNIndex(decoderResult, number)
        // Each sub-word in the loop is a pair. First element representing the index and second element representing its probability
        for (subWord in topSubWords) {
            val nextDecoderInput = createDecoderInputWithNewToken(
                decoderInput,
                subWord.first,
                itr
            )
            // Add the starting of each word in the words ArrayList
            // Increment ITR because we have invoked decoder at line 167
            words.add(Triple(nextDecoderInput, prob * subWord.second, itr+1))
        }

        // Here the words length is same as the number argument of the function
        for (idx in words.indices) {
            // Get first subword for the word at idx
            var nextSubWord = topSubWords[idx].first
            // Keep finding the next subword token till we encounter end of sentence (EOS) token
            // We will also break the loop when we encounter a space (logic inside while loop)
            while (nextSubWord != 2 &&
                        words[idx].third < 27
            ) {
                // Decode the predicted sub word
                val decodedSubWord = subWordDecoder[nextSubWord]!!
                // If the next predicted sub word contains "space" then break as the word has finished
                if (decodedSubWord.contains(wordSeparatorAscii.toChar())) break

                val decoderResult = decoder.run(encoderResult, words[idx].first, words[idx].third)
                val topSubWord = findTopNIndex(decoderResult, 1)
                nextSubWord = topSubWord[0].first
                // Update the decoder input with the new sub word that was predicted
                // Update the probability by aggregating (multiplying) the prob of new generated token
                // Third element is the increased itr
                words[idx] = Triple(createDecoderInputWithNewToken(words[idx].first, nextSubWord, words[idx].third), words[idx].second * topSubWord[0].second, words[idx].third+1)
            }
        }

        return words
    }
}
