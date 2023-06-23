package com.karyaplatform.karya

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.TextView
import androidx.core.widget.addTextChangedListener
import com.karyaplatform.karya.inmt.AssistiveEditText
import com.karyaplatform.karya.inmt.SuggestionAdapter
import com.karyaplatform.karya.inmt.enum.MODE
import com.karyaplatform.karya.inmt.inmtModel.NgINMTModel
import com.nex3z.flowlayout.FlowLayout

class TranslationActivity : AppCompatActivity() {

    private lateinit var translationMode: MODE

    // View elements
    private lateinit var sourceSentenceEt: EditText
    private lateinit var targetSentenceEt: AssistiveEditText
    private lateinit var assistanceFl: FlowLayout

    // model
    private lateinit var model: NgINMTModel

    // static files paths
    private val VOCAB_FILE_PATH = "vocab_tfb_en_hi.json"
    private val SOURCE_SPIECE_PATH = "source_tfb_en_hi.spm"
    private val TARGET_SPIECE_PATH = "target_tfb_en_hi.spm"
    private val ENCODER_FILE_NAME = "tfb_en_hi_quantized_encoder.tflite"
    private val DECODER_FILE_NAME = "tfb_en_hi_quantized_decoder.tflite"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_translation)

        // Get intent and decide the translation mode
        val modeString = intent.getStringExtra("mode")
        translationMode = enumValueOf(modeString!!)

        // Set value of views
        sourceSentenceEt = findViewById(R.id.sourceSentenceEt)
        targetSentenceEt = findViewById(R.id.targetSentenceEt)
        assistanceFl = findViewById(R.id.assistanceFl)

        targetSentenceEt.setSourceSentence(sourceSentenceEt.text.toString())

        sourceSentenceEt.addTextChangedListener { input ->
            val text = input.toString()
            targetSentenceEt.setSourceSentence(text)
        }

        // Set up INMT model
        model = initialiseINMTModel()

        when (translationMode) {
            MODE.DROPDOWN -> {
                setUpDropdown()
            }
            MODE.BOW -> {
                setUpDynamicBOW()
            }
            else -> {}
        }
    }

    private fun setUpDropdown() {
        val adapter = SuggestionAdapter(this, targetSentenceEt, model, MODE.DROPDOWN)
        targetSentenceEt.setAdapter(adapter)
    }

    private fun setUpDynamicBOW() {
        val adapter = SuggestionAdapter(this, targetSentenceEt, model, MODE.BOW) { _: CharSequence?, result: ArrayList<String> ->
            // Determine current words that are already typed by the user
            val currentWords = targetSentenceEt.text.toString().split(" ")
            // Clear the flowlayout
            assistanceFl.removeAllViews()
            for (sentence in result) {
                // Mask first few words already typed by users from suggestion
                val typed = targetSentenceEt.text.toString() // Remove starting and trailing whitespace
                val typedWords = typed.split(" ")
                val partialWord = typedWords[typedWords.size-1]
                val maskLength = typed.split(" ").size - 1
                val words = sentence.split(" ")
                val bowWords = words.slice(IntRange(maskLength, words.size-1))
                for (word in bowWords) {
                    if (word.isEmpty() || word in currentWords) continue
                    val button = createBagOfWordsButton(word)
                    assistanceFl.addView(button)
                }
            }
            assistanceFl.invalidate()
        }
        targetSentenceEt.setAdapter(adapter)
    }

    private fun createBagOfWordsButton(word: String): View? {
        val button = layoutInflater.inflate(R.layout.item_float_bag_of_words, null)
        val wordTv = button.findViewById<TextView>(R.id.wordTv)
        wordTv.text = word
        button.setOnClickListener {
            targetSentenceEt.setText(targetSentenceEt.text.toString().trim() + " " + word)
            targetSentenceEt.setSelection(targetSentenceEt.length()) // placing cursor at the end of the text
        }
        return button
    }

    private fun initialiseINMTModel(): NgINMTModel {
        val vocabStream = assets.open(VOCAB_FILE_PATH)
        val sourceSpieceStream = assets.open(SOURCE_SPIECE_PATH)
        val targetSpieceStream = assets.open(TARGET_SPIECE_PATH)
        val encoderDescriptor = assets.openFd(ENCODER_FILE_NAME)
        val decoderDescriptor = assets.openFd(DECODER_FILE_NAME)
        return NgINMTModel(vocabStream, sourceSpieceStream, targetSpieceStream, encoderDescriptor, decoderDescriptor)
    }

}