import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object MileagePredictor_TEST {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Car Mileage Predictor Test")
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
    val outFile = new BufferedWriter(new FileWriter("output/MileagePredictor_TEST_" + xt + ".txt"))

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

    /** Test with the selected best model */

    val c = 0.01

    val selected = new LinearRegression().setLabelCol("mpg").setMaxIter(100).setRegParam(c).setElasticNetParam(0.6)

    val finalModel = selected.fit(train)

    outFile.append(s"Final model with Reg: ${c}\nCoefficients: ${finalModel.coefficients}\nIntercept: ${finalModel.intercept}\n")
    println(s"Final model with Reg: ${c}\nCoefficients: ${finalModel.coefficients}\nIntercept: ${finalModel.intercept}")

    val training_summary = finalModel.summary
    outFile.append(s"numIterations: ${training_summary.totalIterations}\n")
    println(s"numIterations: ${training_summary.totalIterations}")
    outFile.append(s"Iteration Summary History: ${training_summary.objectiveHistory.toList}\n")
    println(s"Iteration Summary History: ${training_summary.objectiveHistory.toList}")
    outFile.append(s"RMSE: ${training_summary.rootMeanSquaredError}\n")
    println(s"RMSE: ${training_summary.rootMeanSquaredError}")
    outFile.append(s"r2: ${training_summary.r2}\n")
    println(s"r2: ${training_summary.r2}")

    //training_summary.residuals.rdd.coalesce(1).map(_.toString()).saveAsTextFile("output")

    val prediction = finalModel.transform(test)

    val testEvaluator = new RegressionEvaluator().setLabelCol("mpg")
    val testRmse = testEvaluator.evaluate(prediction)
    val testMse = testEvaluator.setMetricName("mse").evaluate(prediction)
    outFile.append("Evaluation on Test set is RMSE: " + "%6.3f".format(testRmse) + " MSE: " + "%6.3f".format(testMse) + "\n")
    println("Evaluation on Test set is RMSE: " + "%6.3f".format(testRmse) + " MSE: " + "%6.3f".format(testMse))

    outFile.close()

    spark.stop()
  }

}