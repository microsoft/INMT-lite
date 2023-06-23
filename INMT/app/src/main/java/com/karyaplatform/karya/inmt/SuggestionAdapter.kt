package com.karyaplatform.karya.inmt

import android.content.Context
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Filter
import android.widget.TextView
import com.karyaplatform.karya.R
import com.karyaplatform.karya.inmt.inmtModel.NgINMTModel
import com.karyaplatform.karya.inmt.enum.MODE
import kotlin.collections.ArrayList

/**
 * The class sets up the suggestion adapter to show suggestions to the user while using AssistiveAutocomplete textview with INMT model
 * @param context
 * @param model: Instance of inmt model
 * @param onPublishResult: Callback that has content of edit text as first argument (constraint) and inmt model suggestions as its second argument (result)
 */
class SuggestionAdapter(
    context: Context,
    private val suggestionEditText: AssistiveEditText,
    private val model: NgINMTModel,
    private val mode: MODE,
    private val onPublishResult: ((constraint: CharSequence?, result: ArrayList<String>) -> Unit)? = null
) : ArrayAdapter<String>(context, 0) {

    val triggerInterval = suggestionEditText.getTriggerAfterEvery()
    val forwardCount = suggestionEditText.getForwardCount()
    val depthCount = suggestionEditText.getDepthCount()

    var suggestions: MutableList<String> = ArrayList()
    private val viewInflater =
        context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        var view: View? = convertView
        if (convertView == null) {
            view = viewInflater.inflate(R.layout.inmt_dropdown_item, null)
        }
        val suggestion: String = getItem(position)!!
        val textView = view!!.findViewById<TextView>(R.id.itemTv)
        textView.text = suggestion
        return view
    }

    override fun getFilter(): Filter {
        return suggestionFilter
    }

    /**
     * Custom Filter implementation for custom suggestions we provide.
     */
    private var suggestionFilter: Filter = object : Filter() {
        override fun convertResultToString(resultValue: Any): CharSequence {
            return resultValue as String
        }

        override fun performFiltering(constraint: CharSequence?): FilterResults {
            Log.i("ADAPTER STRING: ", constraint.toString())
            val input = constraint.toString()
            // Do not invoke model if user is still completing a word
            // TODO: Put the word seperator (space) as a variable
            val tokens = input.split(" ")
            // Check if we are invoking model according to the threshold provided by the view
            if (tokens[tokens.size - 1] != "" ||
                (tokens.size - 1) % triggerInterval != 0
            ) {
                val partialLastWord = tokens[tokens.size - 1]
                val filterResults = FilterResults()
                val filteredResults =
                    suggestions.filter { suggestion -> suggestion.contains(partialLastWord) }
                filterResults.values = filteredResults
                filterResults.count = filteredResults.size
                return filterResults
            }
            return if (constraint != null) {
                suggestions.clear()
                val startRunTs = System.currentTimeMillis()
                val results = model.run(
                    suggestionEditText.getSourceSentence(),
                    constraint.toString(),
                    forwardCount,
                    depthCount
                )
                Log.i("TOTAL_INFERENCE_TIME", (System.currentTimeMillis() - startRunTs).toString())
                for (suggestion in results) {
                    suggestions.add(suggestion)
                }
                val filterResults = FilterResults()
                filterResults.values = suggestions
                filterResults.count = suggestions.size
                filterResults
            } else {
                FilterResults()
            }
        }

        override fun publishResults(constraint: CharSequence?, results: FilterResults?) {
            if (mode == MODE.DROPDOWN) {
                if (results != null && results.count > 0) {
                    clear()
                    for (suggestion in results.values as ArrayList<String>) {
                        add(suggestion)
                        notifyDataSetChanged()
                    }
                }
            } else {
                if (onPublishResult == null) {
                    throw Error("Please provide non null value for onPublishResult argument when mode is not DropDown")
                }
                if (results != null) {
                    onPublishResult!!(constraint, results.values as ArrayList<String>)
                }
            }
        }
    }
}