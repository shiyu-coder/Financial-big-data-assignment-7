package test;

public class IrisClassification {
    public static void main(String[] args){
        String input = args[0];
        String output = args[1];
        String test_data_path = args[2];
        String disFun = "Euclidean";
//        Euclidean, Chebyshev, Manhattan
        try{
            KNN.run(input, output, test_data_path, disFun);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
