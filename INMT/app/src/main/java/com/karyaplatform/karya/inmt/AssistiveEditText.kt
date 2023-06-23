package com.karyaplatform.karya.inmt

import android.content.Context
import androidx.appcompat.widget.AppCompatAutoCompleteTextView
import android.util.AttributeSet
import com.karyaplatform.karya.R
import com.karyaplatform.karya.inmt.enum.MODE
import kotlin.properties.Delegates

class AssistiveEditText(context: Context, attrs: AttributeSet) : AppCompatAutoCompleteTextView(context, attrs) {

    private lateinit var mMode: MODE
    private var mForward by Delegates.notNull<Int>()
    private var mDepth by Delegates.notNull<Int>()
    private var mTriggerAfterEvery by Delegates.notNull<Int>()
    private var mSourceSentence: String? = null


    // Initialise the attributes from XML
    init {
        System.loadLibrary("android_sentencepiece");
        context.theme.obtainStyledAttributes(
            attrs,
            R.styleable.AssistiveEditText,
            0, 0).apply {

            try {
                // TODO: Find a way to convert int to enum
                setSourceSentence(getString(R.styleable.AssistiveEditText_source_sentence))
                setForwardCount(getInteger(R.styleable.AssistiveEditText_forward, 0))
                setDepthCount(getInteger(R.styleable.AssistiveEditText_depth, 0))
                setTriggerAfterEvery(getInteger(R.styleable.AssistiveEditText_trigger_after_every, 0))
            } finally {
                recycle()
            }
        }
    }

    fun getSourceSentence(): String {
        if (mSourceSentence == null) {
            throw IllegalAccessError("Sentence was never set in the Assistive edit text")
        }
        return mSourceSentence!!
    }

    fun getForwardCount(): Int {
        return mForward
    }

    fun getDepthCount(): Int {
        return mDepth
    }

    fun getTriggerAfterEvery(): Int {
        return mTriggerAfterEvery
    }

    fun setMode(mode: MODE) {
        mMode = mode
    }

    fun setSourceSentence(sentence: String?) {
        mSourceSentence = sentence
    }

    fun setForwardCount(forward: Int) {
        if (forward == 0) {
            throw Error("Attribute forward cannot be empty or zero")
        }
        mForward = forward
    }

    fun setDepthCount(depth: Int) {
        if (depth == 0) {
            throw Error("Attribute depth cannot be empty or zero")
        }
        mDepth = depth
    }

    fun setTriggerAfterEvery(triggerAfterEvery: Int) {
        if (triggerAfterEvery == 0) {
            throw Error("Attribute triggerAfterEvery cannot be empty or zero")
        }
        mTriggerAfterEvery = triggerAfterEvery
    }

}