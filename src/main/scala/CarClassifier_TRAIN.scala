import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

//import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
//import org.apache.spark.ml.classification.DecisionTreeClassifier

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.FeatureHasher // Feature Extractor

import org.apache.spark.sql.SparkSession

object CarClassifier_TRAIN {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Car Classifier Train")
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
    val outFile = new BufferedWriter(new FileWriter("output/CarClassifier_TRAIN_" + xt + ".txt"))

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

    /** Learn models and check accuracy */

    // Create list of regularization values for the model to be trained with
    val regList = List(0.001, 0.01, 0.1, 1.0, 10)
    var accMap: Map[Double, Double] = Map()

    for (c <- regList) {

      /** Choose Classifier */
      val classifier = new LogisticRegression().setFeaturesCol("features").setLabelCol("automatic").setMaxIter(100).setRegParam(c).setElasticNetParam(0.6)
      //val classifier = new LinearSVC().setFeaturesCol("features").setLabelCol("automatic").setMaxIter(100).setRegParam(c)
      //val classifier = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("automatic")

      /** Train model */
      val model = classifier.fit(train)

      /** Validate model */
      val result = model.transform(vali)
      //result.select("automatic", "prediction").show()

      /** Check accuracy */
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol("automatic").setMetricName("accuracy")
      val accuracy = evaluator.evaluate(result)

      accMap += (c -> accuracy)

    }

    for ((c, accuracy) <- accMap){
      outFile.append("Reg: " + c + " RMSE: " + "%6.3f".format(accuracy) + "\n")
      println("Reg: " + c + " RMSE: " + "%6.3f".format(accuracy))
    }

    outFile.append("Accuracy is maximum for " + accMap.maxBy(_._2).toString())
    println("Accuracy is maximum for " + accMap.maxBy(_._2).toString())

    outFile.close()

    spark.stop()

    /** Select best model, that has minimum error / maximum accuracy
      * Run the TEST program with the selected best model */

    /** Test with best model */
  }

}