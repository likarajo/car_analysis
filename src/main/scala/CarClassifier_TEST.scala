import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

//import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
//import org.apache.spark.ml.classification.DecisionTreeClassifier

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.sql.SparkSession

object CarClassifier_TEST {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Car Classifier Test")
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
    val outFile = new BufferedWriter(new FileWriter("output/CarClassifier_TEST_" + xt + ".txt"))

    // Create DataFrame using the data file
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(dataFile)

    /** Create a feature vector from given features, that can be passed to learning algorithm */
    val hasher = new FeatureHasher()
      .setInputCols(Array("mpg", "displacement", "hp", "torque", "CRatio", "RARatio", "CarbBarrells", "NoOfSpeed", "length", "width", "weight"))
      .setOutputCol("features")
    val featurized = hasher.transform(df)

    /** Split data into training(60%), validation(20%), and testing(20%) sets */
    val Array(train, vali, test) = featurized.randomSplit(Array(0.6, 0.2, 0.2))

    /** Test with best model */

    val c = 0.1

    val selected = new LogisticRegression().setFeaturesCol("features").setLabelCol("automatic").setMaxIter(100).setRegParam(c).setElasticNetParam(0.6)
    //val selected = new LinearSVC().setFeaturesCol("features").setLabelCol("automatic").setMaxIter(100).setRegParam(c)
    //val selected = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("automatic")
    //selected.toDebugString .. only for Decision Tree

    val finalModel = selected.fit(train)

    //Use only for Logistic Regression
    outFile.append(s"Final model\nCoefficients: ${finalModel.coefficients}\nIntercept: ${finalModel.intercept}\n")
    println(s"Final model\nCoefficients: ${finalModel.coefficients}\nIntercept: ${finalModel.intercept}")

    val prediction = finalModel.transform(test)

    val testAccuracy = new MulticlassClassificationEvaluator().setLabelCol("automatic").setMetricName("accuracy").evaluate(prediction)

    outFile.append(s"Accuracy on Test set is: ${testAccuracy}\n")
    println(s"Accuracy on Test set is: ${testAccuracy}")

    outFile.close()

    spark.stop()
  }

}