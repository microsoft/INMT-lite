package com.example.inmt_offline.Adapters;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.example.inmt_offline.R;

import java.util.ArrayList;

public class PreviewListAdapter extends RecyclerView.Adapter<PreviewListAdapter.MyViewHolder> {
    private ArrayList<String> qTrans;
    private ArrayList<String> aTrans;

    // Provide a reference to the views for each data item
    // Complex data items may need more than one view per item, and
    // you provide access to all the views for a data item in a view holder
    public static class MyViewHolder extends RecyclerView.ViewHolder {
        // each data item is just a string in this case
        public TextView qTranstextView;
        public TextView aTranstextView;
        public MyViewHolder(View v) {
            super(v);
            qTranstextView = v.findViewById(R.id.qTransTextView);
            aTranstextView = v.findViewById(R.id.aTransTextView);
        }
    }

    // Provide a suitable constructor (depends on the kind of dataset)
    public PreviewListAdapter(ArrayList<String> qTrans, ArrayList<String> aTrans) {
        this.qTrans = qTrans;
        this.aTrans = aTrans;
    }

    // Create new views (invoked by the layout manager)
    @Override
    public PreviewListAdapter.MyViewHolder onCreateViewHolder(ViewGroup parent,
                                                              int viewType) {
        // create a new view
        View previewCard = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.preview_card, parent, false);
        MyViewHolder vh = new MyViewHolder(previewCard);
        return vh;
    }

    // Replace the contents of a view (invoked by the layout manager)
    @Override
    public void onBindViewHolder(MyViewHolder holder, int position) {
        // - get element from your dataset at this position
        // - replace the contents of the view with that element
        holder.qTranstextView.setText(qTrans.get(position));
        holder.aTranstextView.setText(aTrans.get(position));

    }

    // Return the size of your dataset (invoked by the layout manager)
    @Override
    public int getItemCount() {
        return qTrans.size();
    }
}