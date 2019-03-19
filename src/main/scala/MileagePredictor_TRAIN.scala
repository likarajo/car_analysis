import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object MileagePredictor_TRAIN {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Car Mileage Predictor Train")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    // Get current time
    val xt = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYMMddHHmmss"))

    // Specify data file
    val dataFile = "data/car-mileage.csv"

    // Specify output file
    val outFile = new BufferedWriter(new FileWriter("output/MileagePredictor_TRAIN_" + xt + ".txt"))

    // Create DataFrame using the data file
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(dataFile)
      .na.drop() // remove rows with null or NaN values

    /** Create a vector from given features, that can be passed to learning algorithm */
    val assembler = new VectorAssembler()
      .setInputCols(Array("displacement", "hp", "torque", "CRatio", "RARatio", "CarbBarrells", "NoOfSpeed", "length", "width", "weight", "automatic"))
      .setOutputCol("features")
    val assembled = assembler.transform(df)

    /** Split data into training(60%), validation(20%), and testing(20%) sets */
    val Array(train, vali, test) = assembled.randomSplit(Array(0.6, 0.2, 0.2))

    /** Learn models and check accuracy */

    // Create list of regularization values for the model to be trained with
    val regList = List(0.001, 0.01, 0.1, 1.0, 10)
    var regMap: Map[Double, (Double, Double)] = Map()

    for (c <- regList) {

      /** Choose Classifier */
      val classifier = new LinearRegression()
        .setLabelCol("mpg").setMaxIter(100).setRegParam(c).setElasticNetParam(0.6)

      /** Train model */
      val model = classifier.fit(train)

      /** Validate model */
      val result = model.transform(vali)
      //result.show()

      /** Check accuracy */
      val evaluator = new RegressionEvaluator().setLabelCol("mpg")
      val rmse = evaluator.evaluate(result)
      val mse = evaluator.setMetricName("mse").evaluate(result)
      regMap += (c -> (rmse, mse))

    }

    for ((c, (rmse, mse)) <- regMap){
      outFile.append("Reg: " + c + " RMSE: " + "%6.3f".format(rmse) + " MSE: " + "%6.3f".format(mse) + "\n")
      println("Reg: " + c + " RMSE: " + "%6.3f".format(rmse) + " MSE: " + "%6.3f".format(mse) + "\n")
    }

    outFile.append("RMSE and MSE are minimum for " + regMap.minBy(_._2._1).toString())
    println("RMSE and MSE are minimum for " + regMap.minBy(_._2._1).toString())

    outFile.close()

    spark.stop()

    /** Select best model, that has minimum error / maximum accuracy
      * Run the TEST program with the selected best model */

    /** Test with best model */
  }

}