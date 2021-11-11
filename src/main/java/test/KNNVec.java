package test;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class KNNVec implements WritableComparable<KNNVec> {
    Integer id;
    Double dis;

    public KNNVec(){}

    public KNNVec(Integer id, Double dis){
        super();
        this.id = id;
        this.dis = dis;
    }

    @Override
    public void write(DataOutput out) throws IOException{
        out.writeInt(id);
        out.writeDouble(dis);
    }

    @Override
    public void readFields(DataInput in) throws IOException{
        id = in.readInt();
        dis = in.readDouble();
    }

    @Override
    public int compareTo(KNNVec v){
        if(v.id.equals(id))
            return dis.compareTo(v.dis);
        return id.compareTo(v.id);
    }

    @Override
    public boolean equals(Object o){
        KNNVec v = (KNNVec)o;
        return v.compareTo(this)==0;
    }

    @Override
    public int hashCode(){
        return id.hashCode();
    }

    @Override
    public String toString(){
        return id.toString()+":"+dis.toString();
    }

}
