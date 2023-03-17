import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Application {
    public static void main(String[] args) {


        SparkSession sparkSession=SparkSession.builder().appName("spark-mllib").master("local").getOrCreate();

        Dataset<Row> raw_data = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("C:\\Users\\fulli\\Desktop\\sampledata");

        VectorAssembler features_vector=new VectorAssembler().setInputCols(new String[]{"Ay"})
                .setOutputCol("features");

        Dataset<Row> transform = features_vector.transform(raw_data);
        Dataset<Row> finalData = transform.select("features", "Satis");

        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7, 0.3}); // Test And Train
        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData = datasets[1];

        LinearRegression lr=new LinearRegression();
        lr.setLabelCol("Satis");

        LinearRegressionModel model = lr.fit(trainData);  // Creating Model
        Dataset<Row> transformTest = model.transform(testData);
        transformTest.show();

    }
}
