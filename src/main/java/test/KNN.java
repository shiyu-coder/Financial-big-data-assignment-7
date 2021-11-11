package test;

import com.jcraft.jsch.IO;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class KNN {

    private static String test_data;
    private static String test_label;

    public static class KNNMap extends Mapper<LongWritable, Text, KNNVec, Text>{
        static Integer id;
        private ArrayList<String> testData = new ArrayList<String>();
        String disFun;

        @Override
        protected void setup(Mapper<LongWritable, Text, KNNVec, Text>.Context context) {
            id = 1;
            Configuration conf = context.getConfiguration();
            disFun = conf.get("disFun", "Euclidean");
            testData = KNN.readFile(test_data);
//            try{
//                Path[] cacheFiles = context.getLocalCacheFiles();
//                if(cacheFiles != null && cacheFiles.length > 0){
//                    BufferedReader reader = new BufferedReader(new FileReader(cacheFiles[0].toString()));
//                    try{
//                        String line = null;
//                        while((line = reader.readLine())!= null){
//                            testData.add(line);
//                        }
//                    }finally {
//                        reader.close();
//                    }
//                    super.setup(context);
//                }
//            }catch (Exception e){
//                e.printStackTrace();
//            }

        }

        @Override
        protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, KNNVec, Text>.Context context) {
            String train_data = value.toString();
            String[] features = train_data.split(",");
            String label = features[features.length - 1];
            for(String i: testData){
                String[] f = i.split(",");
                Double dis = getDistance(features, f, disFun);
                try{
                    context.write(new KNNVec(id, dis), new Text(label));
                }catch (Exception e){
                    e.printStackTrace();
                }
                id++;
            }
            id = 1;
        }

        private Double getDistance(String[] f1, String[] f2, String disFun){
            Double dis = 0.0;
            switch (disFun){
                case "Euclidean":
                    for(int i=0;i<f2.length;i++){
                        dis += Math.pow(Double.parseDouble(f1[i]) - Double.parseDouble(f2[i]), 2);
                    }
                    dis = Math.sqrt(dis);
                    break;
                case "Chebyshev":
                    for(int i=0;i<f2.length;i++){
                        dis = Math.max(Double.parseDouble(f1[i]) - Double.parseDouble(f2[i]), dis);
                    }
                    break;
                case "Manhattan":
                    for(int i=0;i<f2.length;i++){
                        dis = Math.abs(Double.parseDouble(f1[i]) - Double.parseDouble(f2[i]));
                    }
                    break;
            }
            return dis;
        }
    }

    public static class KNNCombiner extends Reducer<KNNVec, Text, KNNVec, Text>{
        static int k;
        int j;
        Integer current_id;

        public void setup(Context context){
            Configuration conf = context.getConfiguration();
            k = conf.getInt("k", 3);
            current_id = null;
            j = k;
        }

        @Override
        protected void reduce(KNNVec key, Iterable<Text> value, Reducer<KNNVec, Text, KNNVec, Text>.Context context)
            throws IOException, InterruptedException{
            if(current_id == null){
                current_id = key.id;
            }
            if(current_id != key.id){
                j = k;
                current_id = key.id;
            }
            if(j != 0){
                for(Text i: value){
                    j--;
                    context.write(key, i);
                    if(j == 0)
                        break;
                }
            }
        }
    }

    public static class KNNReduce extends Reducer<KNNVec, Text, Text, NullWritable> {
        static int k;
        String disFun;
        int j;
        HashMap<String, Integer> results;
        Integer current_id;
        double right_count;
        double false_count;
        private ArrayList<String> labels = new ArrayList<String>();

        public void setup(Context context){
            Configuration conf = context.getConfiguration();
            k = conf.getInt("k", 3);
            disFun = conf.get("disFun", "Euclidean");
            results = new HashMap<String, Integer>();
            current_id = null;
            j = k;

            right_count = 0;
            false_count = 0;
            labels = KNN.readFile(KNN.test_label);
//            try{
//                Path[] cacheFiles = context.getLocalCacheFiles();
//                if (cacheFiles != null && cacheFiles.length > 0){
//                    BufferedReader reader = new BufferedReader(new FileReader(cacheFiles[1].toString()));
//                    try{
//                        String line = null;
//                        while((line = reader.readLine()) != null){
//                            labels.add(line.trim());
//                        }
//                    }finally {
//                        reader.close();
//                    }
//                    super.setup(context);
//                }
//            }catch (Exception e){
//                e.printStackTrace();
//            }

        }

        private String getResult(HashMap<String, Integer> map){
            String result = null;
            Integer max_count = 0;
            Iterator iter = map.entrySet().iterator();
            while (iter.hasNext()) {
                Map.Entry entry = (Map.Entry) iter.next();
                String key = (String)entry.getKey();
                Integer val = (Integer)entry.getValue();
                if(val > max_count){
                    result = key;
                    max_count = val;
                }
            }
            return result;
        }

        @Override
        protected void reduce(KNNVec key, Iterable<Text> value, Reducer<KNNVec, Text, Text, NullWritable>.Context context){
            if(current_id == null){
                current_id = key.id;
            }
            if(!current_id.equals(key.id)){
                j = k;
                current_id = key.id;
            }
            if(j != 0){
                for(Text i: value){
                    Integer count = 1;
                    if(results.containsKey(i.toString()))
                        count += results.get(i.toString());
                    results.put(i.toString(), count);
                    j--;
                    if(j == 0){
                        String newKey = getResult(results).trim();
                        try{
                            context.write(new Text(newKey), NullWritable.get());
                        }catch (Exception e){
                            e.printStackTrace();
                        }

                        results.clear();
                        if(labels.get((int)(false_count + right_count)).equals(newKey))
                            right_count += 1;
                        else
                            false_count += 1;
                    }
                }
            }
        }

        @Override
        protected void cleanup(Reducer<KNNVec, Text, Text, NullWritable>.Context context)
            throws IOException, InterruptedException{
            context.write(new Text("距离计算: "+disFun), NullWritable.get());
            context.write(new Text("类别数: "+k), NullWritable.get());
            context.write(new Text("准确率accuracy: "+right_count/(right_count+false_count)), NullWritable.get());

        }
    }

//    private static ArrayList<String> readFile(String file_path) {
//        ArrayList<String> cont = new ArrayList<String>();
//        try {
//            Path file = new Path(file_path);
//            BufferedReader fis = new BufferedReader(new FileReader(file.toString()));
//            String line = null;
//            while ((line = fis.readLine()) != null) {
//                cont.add(line);
//            }
//        } catch (IOException ioe) {
//            System.err.println("Exception while reading stop word file '"
//                    + cont + "' : " + ioe.toString());
//        }
//        return cont;
//    }

    private static ArrayList<String> readFile(String file_path) {
        ArrayList<String> cont = new ArrayList<String>();
        Configuration conf = new Configuration();
        try{
            System.out.println(file_path);
            FileSystem fs = FileSystem.get(URI.create(file_path),conf);
            InputStream is = fs.open(new Path(file_path));
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line = null;
            while ((line = reader.readLine()) != null) {
                cont.add(line);
            }
        }catch (IOException ioe){
            ioe.printStackTrace();
        }
        return cont;
    }

    public static void run(String input, String output, String test_data_path, String disFun){
        try{
            Configuration conf = new Configuration();
            conf.set("disFun", disFun);
            Job job = Job.getInstance(conf, "Iris Classification - KNN");
            job.setJarByClass(KNN.class);
            job.setMapperClass(KNNMap.class);
            job.setCombinerClass(KNNCombiner.class);
            job.setReducerClass(KNNReduce.class);
            job.setMapOutputKeyClass(KNNVec.class);

            test_data = test_data_path + "test/test_data.csv";
            test_label = test_data_path + "test/test_label.csv";

//        job.addCacheFile(new Path(test_data_path + "test/test_data.csv").toUri());
//        job.addCacheFile(new Path(test_data_path + "test/test_label.csv").toUri());

            FileInputFormat.addInputPath(job, new Path(input));
            FileOutputFormat.setOutputPath(job, new Path(output));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }catch (Exception e){
            e.printStackTrace();
        }

    }

}
