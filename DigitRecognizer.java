/* 
Implements a fully connected feed forward network using stochastic gradient descent back propogation algorithms to predict the values of digits.
*/ 

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.List;
import java.util.Collections;

public class DigitRecognizer {
    public static final int MINI_BATCH_SIZE = 10;
    public static final double ETA = 3;
    public static final int NUMBER_OF_EPOCHS = 5;
    public static final String  TESTING_DATA_FILE_NAME = "mnist_test.csv";
    public static final String  TRAINING_DATA_FILE_NAME = "mnist_train.csv";
    public static final String INPUT_WEIGHTS_FILE_NAME = "weights.csv";
    public static final String INPUT_BIASES_FILE_NAME = "biases.csv";
    public static final String OUTPUT_WEIGHTS_FILE_NAME = "output_weights.csv";
    public static final String OUTPUT_BIASES_FILE_NAME = "output_biases.csv";
    public static int[][] ACCURACY_DATA = new int[2][10];
    public static int[][] TESTING_SUCCESS = new int[10000][3];
    
    public static void main(String[] args) {
        mainLoop(); 
    }

    public static void resetAccuracyData() {
        ACCURACY_DATA = new int[2][10];
    }

    public static void runTestingCases(Matrix[] testData, Matrix[] correctOutputs, Matrix[] weights, Matrix[] biases, boolean isTesting) {
        boolean success = true;
        int correctOutput;
        for (int i=0; i< testData.length; i++) {
            correctOutput =  Matrix.oneHotToInt(correctOutputs[i]);
            success = runTestingCase(weights, biases, testData[i], correctOutput, i, isTesting);
            // record whether this case was a success
            if (isTesting) {
                TESTING_SUCCESS[i][2] = success ? 1 : 0;
            }
        }
    }

    public static boolean runTestingCase(Matrix[] weights, Matrix[] biases, Matrix trainingData, int correctOutput, int position, boolean isTesting) {
        ACCURACY_DATA[1][correctOutput]++;
        int numLayers = weights.length;

        // matrix arrays for new values
        Matrix[] activations = new Matrix[numLayers+1];

        activations[0] = trainingData;

        // get all activation values by making forward passes
        for (int i=0; i<numLayers; i++) {
            activations[i+1] = Matrix.sigmaFunction(Matrix.add(Matrix.dotProduct(weights[i], activations[i]), biases[i]));
            
        }

        // find position of max value
        double[] values = Matrix.transpose(activations[numLayers]).getRow(0);
        int maxValuePosition = 0;
        double maxValue = values[0];
        for (int i=1; i<values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxValuePosition = i;
            }
        }

        // record correct and output values if testing
        if (isTesting) {
            TESTING_SUCCESS[position][0] = correctOutput;
            TESTING_SUCCESS[position][1] = maxValuePosition;
        }


        if (maxValuePosition == correctOutput) {
            ACCURACY_DATA[0][correctOutput]++;
            return true;
        }
        return false;
    }

    public static Matrix[][] runTrainingCase(Matrix[] weightsMatrices, Matrix[] biasesMatrices, Matrix trainingData, Matrix correctOutputs) {
        int numLayers = weightsMatrices.length;

        // matrix arrays for new values
        Matrix[] activations = new Matrix[numLayers+1];
        Matrix[] gradientBiases = new Matrix[numLayers];
        Matrix[] gradientWeights = new Matrix[numLayers];

        activations[0] = trainingData;

        // get all activation values by making forward passes
        for (int i=0; i<numLayers; i++) {
            activations[i+1] = Train.forwardPass(weightsMatrices[i],activations[i],biasesMatrices[i]);
        }

        // get gradient bias for final layer
        gradientBiases[numLayers-1] = Train.getGBFinalLayer(activations[numLayers], correctOutputs);

        // get gradient biases for all non-final layers
        for (int i=1; i<numLayers; i++) {
            gradientBiases[numLayers-i-1] = Train.getGBNonFinalLayer(weightsMatrices[numLayers-i],activations[numLayers-i],gradientBiases[numLayers-i]);
        }

        // get all gradient of weights
        for (int i=0; i<numLayers; i++) {
            gradientWeights[numLayers-i-1] = Train.gradientOfWeights(activations[numLayers-i-1],gradientBiases[numLayers-i-1]);
        }

        Matrix[][] results = {gradientWeights,gradientBiases};

        return results;
    }

    public static Matrix[][] runMiniBatch(Matrix[][] values, Matrix[] trainingData, Matrix[] correctOutputs) {
        int numLayers = values[0].length;

        // break up weights and bias matrices
        Matrix[] weightsMatrices = values[0];
        Matrix[] biasesMatrices = values[1];

        // initialize arrays for the new values
        Matrix[] revisedWeightsMatrices = new Matrix[numLayers];
        Matrix[] revisedBiasesMatrices = new Matrix[numLayers];
        Matrix[][] gradientBiasMatrices = new Matrix[MINI_BATCH_SIZE][numLayers];
        Matrix[][] gradientWeightMatrices = new Matrix[MINI_BATCH_SIZE][numLayers];

        // get weight and bias gradients for entire mini batch
        for (int i=0; i<trainingData.length; i++) {
            Matrix[][] gradients = runTrainingCase(weightsMatrices, biasesMatrices, trainingData[i], correctOutputs[i]);
            gradientWeightMatrices[i] = gradients[0];
            gradientBiasMatrices[i] = gradients[1];
        }

        // get revised weigths andbiases
        for (int layer=0; layer<numLayers; layer++) {
            revisedWeightsMatrices[layer] = Train.reviseValues(weightsMatrices[layer], Train.getLayerValues(gradientWeightMatrices, layer), ETA, MINI_BATCH_SIZE);
            revisedBiasesMatrices[layer] = Train.reviseValues(biasesMatrices[layer], Train.getLayerValues(gradientBiasMatrices, layer), ETA, MINI_BATCH_SIZE);
        }

        Matrix[][] result = {revisedWeightsMatrices, revisedBiasesMatrices};

        return result;
        
    }

    public static Matrix[][] runEpoch(Matrix[][] values, Matrix[] trainingDataMatrices, Matrix[] correctOutputsMatrices) {
        
        int numberOfMiniBatches = trainingDataMatrices.length / MINI_BATCH_SIZE;

        Matrix[] currentTrainingData;
        Matrix[] currentCorrectOutputs;
 
        for (int i=0; i<numberOfMiniBatches; i++) {
            // get training data and correct outputs to use for current mini batch
            currentTrainingData = Arrays.copyOfRange(trainingDataMatrices,i*MINI_BATCH_SIZE, (i+1)*MINI_BATCH_SIZE);
            currentCorrectOutputs = Arrays.copyOfRange(correctOutputsMatrices,i*MINI_BATCH_SIZE, (i+1)*MINI_BATCH_SIZE);

            // get the updated values to be used in next mini batch
            values = runMiniBatch(values, currentTrainingData, currentCorrectOutputs);
        }

        return values;
    }

    public static void printData(int[][] accuracyData) {
        // print accuracy for each digit
        for (int i=0; i<10; i++) {
            System.out.println("Digit " + i + ": " + accuracyData[0][i] + "/" + accuracyData[1][i]);
        }

        // calculate and display total accuracy
        int total = Arrays.stream(accuracyData[1]).sum();
        int totalCorrect = Arrays.stream(accuracyData[0]).sum();
        double percentCorrect = (double) totalCorrect / total * 100;
        System.out.printf("\n Accuracy: %d/%d = %.3f%% \n\n", totalCorrect, total, percentCorrect);
    }

    public static void printMainPage() {
        System.out.println("0. Exit \n1. Train the Network \n2. Load A Pre-Trained Network \n3. Display Network Accuracy on Training Data \n4. Display Network Accuracy on Testing Data \n5. Run Network On Testing Data Showing All Images \n6. Run Network On Testing Data Showing Only Incorrectly Classified Images\n7. Save Network to File");
    }

    public static void mainLoop() {
        Scanner scanner = new Scanner(System.in);
        Matrix[][] network = null;
        int state = -1;
        try{
            while (state != 0) {
                printMainPage();
                state = getInput(scanner);
                switch (state) {
                    case 1:
                        network = trainNetwork();
                        break;
                    case 2:       
                        network = loadNetwork();
                        break;
                    case 3:
                        if (network != null) {
                            runData(network[0], network[1], TRAINING_DATA_FILE_NAME);
                        }
                        break;
                    case 4:
                        if (network != null) {
                            runData(network[0], network[1], TESTING_DATA_FILE_NAME); 
                        }
                        break;
                    case 5:
                        if (network != null) {
                            printAllNumbers(network[0], network[1], scanner);
                        }
                        break;
                    case 6:
                        if (network != null) {
                            printIncorrectNumbers(network[0], network[1], scanner);
                        }
                        break;
                    case 7:
                        if (network != null) {
                            saveNetworkState(network);
                        }
                    default:
                        break;
                }
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        scanner.close();
    }

    public static int getInput(Scanner scanner) {
        int input = -1;
        System.out.print("Enter: ");
       
        if (scanner.hasNextInt()) {
            input = scanner.nextInt();
        } else {
            // consume token to prevent infinite loop
            scanner.next();
        }
        System.out.println();

        // only return valid ints
        return (input > -1 && input < 8) ? input : -1;  
    }

    public static void printNumber(double[] number) {
        // image is 28X28
        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                System.out.print(number[28*i + j] == 0 ? " " : "0");
            }
            System.out.println();
        }
    }

    public static void printAllNumbers(Matrix[] weights, Matrix[] biases, Scanner scanner) {
        try {
            //get all number representations
            Matrix[][] dataFromFile = Input.getTestingData(TESTING_DATA_FILE_NAME);
            Matrix[] pixels = dataFromFile[1];
            runTestingCases(dataFromFile[1],dataFromFile[0],weights,biases, true);

            // print display screen for each number
            for (int i=0; i<pixels.length; i++) {
                System.out.printf("Testing Case #%d: Correct classification = %d Network Output = %d", i, TESTING_SUCCESS[i][0], TESTING_SUCCESS[i][1]);
                printNumber(Matrix.transpose(pixels[i]).getRow(0));
                System.out.println("Enter 0 to quit to main menu. All other values continue.");
                if (getInput(scanner) == 0) {
                    return;
                }
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    public static void printIncorrectNumbers(Matrix[] weights, Matrix[] biases, Scanner scanner) {
        try {
            //get all number representations
            Matrix[][] dataFromFile = Input.getTestingData(TESTING_DATA_FILE_NAME);
            Matrix[] pixels = dataFromFile[1];
            runTestingCases(dataFromFile[1],dataFromFile[0],weights,biases, true);

            for (int i=0; i<pixels.length; i++) {
                // skip successful cases
                if (TESTING_SUCCESS[i][2] == 1) {
                    continue;
                }
                // print display screen for each number
                System.out.printf("Testing Case #%d: Correct classification = %d Network Output = %d", i, TESTING_SUCCESS[i][0], TESTING_SUCCESS[i][1]);
                printNumber(Matrix.transpose(pixels[i]).getRow(0));
                System.out.println("Enter 0 to quit to main menu. All other values continue.");
                if (getInput(scanner) == 0) {
                    return;
                }
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    public static Matrix[][] trainNetwork() throws IOException {
        // get initial matrices
        Matrix[] initialWeightsMatrices = {new Matrix(15,784), new Matrix(10,15)};
        Matrix[] initialBiasesMatrices = {new Matrix(15,1), new Matrix(10,1)};
        Matrix[][] values = {initialWeightsMatrices,initialBiasesMatrices};

        Matrix[][] dataFromFile = Input.getTestingData(TRAINING_DATA_FILE_NAME);
        Matrix[] correctOutputsMatrices = dataFromFile[0];
        Matrix[] trainingDataMatrices = dataFromFile[1];

        // run epochs
        for (int i=0; i < NUMBER_OF_EPOCHS ; i++) {
            values = runEpoch(values,trainingDataMatrices,correctOutputsMatrices);

            // print accuracy at end of epoch
            System.out.println("EPOCH " + (i+1) + " ACCURACY");
            runTestingCases(dataFromFile[1],dataFromFile[0],values[0],values[1], false);
            printData(ACCURACY_DATA);
            resetAccuracyData();
            dataFromFile = Matrix.randomizeArrays(correctOutputsMatrices, trainingDataMatrices);
            correctOutputsMatrices = dataFromFile[0];
            trainingDataMatrices = dataFromFile[1];
        } 
        return values;
    }

    public static void saveNetworkState(Matrix[][] values) throws IOException{
        Input.writeValuesToFile(OUTPUT_WEIGHTS_FILE_NAME, values[0]);
        Input.writeValuesToFile(OUTPUT_BIASES_FILE_NAME, values[1]);
    }

    public static Matrix[][] loadNetwork(){
        Matrix[][] values = null;
        try {
            Matrix[] weights = Input.getValuesFromFile(INPUT_WEIGHTS_FILE_NAME);
            Matrix[] biases = Input.getValuesFromFile(INPUT_BIASES_FILE_NAME);
            values = new Matrix[][]{weights, biases};
        } catch (IOException e) {
            System.out.println("Could not find input files.");
        }
        return values;
    }

    public static void runData(Matrix[] weights, Matrix[] biases, String fileName) throws IOException {
        // get data to test network on
        Matrix[][] dataFromFile = Input.getTestingData(fileName);

        // run testing cases and print accuracy data
        System.out.println("ACCURACY: ");
        runTestingCases(dataFromFile[1],dataFromFile[0],weights,biases, false);
        printData(ACCURACY_DATA);
        resetAccuracyData();
    }

}

class Matrix {
    private double[][] elements;
    private int numRows;
    private int numColumns;

    // initialize matric values
    public Matrix(double[][] elements) {
        this.elements = elements;
        this.numRows = elements.length;
        this.numColumns = elements[0].length;
    }

    public Matrix(double[] elements) {
        double[][] newElements = {elements};
        this.elements = newElements;
        this.numRows = 1;
        this.numColumns = elements.length;
    }

    public Matrix(int numRows, int numColumns) {
        this.numRows = numRows;
        this.numColumns = numColumns;
        this.elements = new double[numRows][numColumns];

        for (int i=0; i<numRows; i++) {
            for (int j=0; j<numColumns; j++) {
                this.elements[i][j] = Math.random() * .01;
            }
        }
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumColumns() {
        return numColumns;
    }

    public double[][] getElements() {
        return elements;
    }

    public double getElement(int row,int column) {
        return elements[row][column];
    }

    public double[] getRow(int row) {
        return elements[row];
    }

    public void setElement(int row, int column, double newValue) {
        elements[row][column] = newValue;
    }

    public static Matrix dotProduct(Matrix m1, Matrix m2) {
        // check for valid dimensions
        if (m1.getNumColumns() != m2.getNumRows()) {
            System.out.println("Invalid matrix dimensions.\n");
            return null;
        }
        // resting matrix will have # rows of the first matrix and # columns of the socond
        Matrix result = new Matrix(new double[m1.getNumRows()][m2.getNumColumns()]);
        Matrix m2Transposed = Matrix.transpose(m2);

        double newElement;

        // take sum product of each row in m1 and each column in m2 to create new matrix
        for (int i=0; i < m1.getNumRows(); i++) {
            for (int j=0; j < m2.getNumColumns(); j++) {
                // row of m2Transposed = column of m2
                newElement = dotProduct(m1.getRow(i),m2Transposed.getRow(j));
                result.setElement(i, j, newElement);
            }
        }

        return result;
    }

    public static double dotProduct(double[] a1, double[] a2) {
        // check for matching lengths
        if (a1.length != a2.length) {
            System.out.println("Invalic array lengths.\n");
            return -1;
        }

        double total = 0;

        // calculate dot product 
        for (int i=0; i < a1.length; i++) {
            total += a1[i] * a2[i];
        }

        return total;
    }

    public static Matrix transpose(Matrix matrix) {
        Matrix newMatrix = new Matrix(new double[matrix.getNumColumns()][matrix.getNumRows()]);
        double newElement;

        // reverse row and column of each element
        for (int i=0; i < matrix.getNumRows(); i++) {
            for (int j=0; j < matrix.getNumColumns(); j++) {
                newElement = matrix.getElement(i,j);
                newMatrix.setElement(j,i,newElement);
            }
        }
        return newMatrix;
    }

    public static Matrix add(Matrix m1, Matrix m2) {
        // ensure like dimensions
        if (m1.getNumColumns() != m2.getNumColumns() || m1.getNumRows() != m2.getNumRows()) {
            System.out.println("Matrices must have same dimensions.");
            return null;
        }

        Matrix result = new Matrix(new double[m1.getNumRows()][m1.getNumColumns()]);
        double newElement;

        // add each respective element in matrices m1 and m2
        for (int i=0; i < m1.getNumRows(); i++) {
            for (int j=0; j < m1.getNumColumns(); j++) {
                newElement = m1.getElement(i, j) + m2.getElement(i, j);
                result.setElement(i, j, newElement);
            }
        }

        return result;
    }

    public static Matrix multiply(Matrix m1, Matrix m2) {
        // ensure like dimensions
        if (m1.getNumColumns() != m2.getNumColumns() || m1.getNumRows() != m2.getNumRows()) {
            System.out.println("Matrices must have same dimensions.");
            return null;
        }
        Matrix result = new Matrix(new double[m1.getNumRows()][m1.getNumColumns()]);
        double newElement;

        // multiply each respective element in m1 and m2
        for (int i=0; i < m1.getNumRows(); i++) {
            for (int j=0; j < m1.getNumColumns(); j++) {
                newElement = m1.getElement(i, j) * m2.getElement(i, j);
                result.setElement(i, j, newElement);
            }
        }
        return result;
    }

    public static Matrix multiplyByConstant(Matrix matrix, double num) {
        Matrix result = new Matrix(new double[matrix.getNumRows()][matrix.getNumColumns()]);
        double newElement;

        // multiply constant by every element in the matrix
        for (int i=0; i < matrix.getNumRows(); i++) {
            for (int j=0; j < matrix.getNumColumns(); j++) {
                newElement = matrix.getElement(i, j) * num;
                result.setElement(i, j, newElement);
            }
        }
        return result;
    }

    public static Matrix addByConstant(Matrix matrix, double num) {
        Matrix result = new Matrix(new double[matrix.getNumRows()][matrix.getNumColumns()]);
        double newElement;

        // add constant by every element in the matrix
        for (int i=0; i < matrix.getNumRows(); i++) {
            for (int j=0; j < matrix.getNumColumns(); j++) {
                newElement = matrix.getElement(i, j) + num;
                result.setElement(i, j, newElement);
            }
        }
        return result;
    }

    public static Matrix sigmaFunction(Matrix matrix) {
        Matrix result = new Matrix(new double[matrix.getNumRows()][matrix.getNumColumns()]);
        double newElement;

        // apply sigma function to each individual element
        for (int i=0; i < matrix.getNumRows(); i++) {
            for (int j=0; j < matrix.getNumColumns(); j++) {
                newElement = sigmaFunction(matrix.getElement(i, j));
                result.setElement(i, j, newElement);
            }
        }

        return result;
    }

    public static double sigmaFunction(double num) {
        // apply definition of the sigma function
        return (1/(1+Math.pow(Math.E,(-1*num))));
    }

    public static int oneHotToInt(Matrix m) {
        if (m.getNumRows() > 1) {
            m = Matrix.transpose(m);
        }
        for (int i=0; i<m.getRow(0).length; i++) {
            if (m.getRow(0)[i] == 1) {
                return i;
            }
        }
        return 0;
    }

    public static Matrix[][] randomizeArrays(Matrix[] m1, Matrix[] m2) {        
        Matrix[] newM1 = new Matrix[m1.length];
        Matrix[] newM2 = new Matrix[m2.length];
        
        // combine the matrix arrays
        List<Matrix[]> pairedList = new ArrayList<>();
        for (int i = 0; i < m1.length; i++) {
            pairedList.add(new Matrix[]{m1[i], m2[i]});
        }
        
        // randomize values
        Collections.shuffle(pairedList);
        
        // split back into distincy arrays
        for (int i = 0; i < pairedList.size(); i++) {
            newM1[i] = pairedList.get(i)[0];
            newM2[i] = pairedList.get(i)[1];
        }

        Matrix[][] finalMatrices = {newM1, newM2};
        
        return finalMatrices;
    }
}

class Train {
    public static Matrix forwardPass(Matrix weights, Matrix activation, Matrix biases) {
        // apply definition of sigma function
        return Matrix.sigmaFunction(Matrix.add(Matrix.dotProduct(weights, activation), biases));
    }

    public static Matrix gradientOfWeights(Matrix activation, Matrix gb) {
        // transpose activation matrix so dimensions match
        Matrix activationTransposed = Matrix.transpose(activation);

        // compute gradient of weights
        Matrix result = Matrix.dotProduct(gb, activationTransposed);

        return result;
    }

    public static Matrix getGBFinalLayer(Matrix activation, Matrix train) {
        // use formula to calculate gb for the final layer
        Matrix derivativeOfActivationFunction = Matrix.multiply(activation, Matrix.addByConstant(Matrix.multiplyByConstant(activation, -1), 1));
        Matrix differenceBetweenCorrectValues = Matrix.add(activation, Matrix.multiplyByConstant(train, -1));
        return Matrix.multiply(derivativeOfActivationFunction,differenceBetweenCorrectValues);
    }

    public static Matrix getGBNonFinalLayer(Matrix weights, Matrix activation, Matrix gb) {
        Matrix result = new Matrix(new double[activation.getNumRows()][1]);
        Matrix tansposedWeights = Matrix.transpose(weights);

        // use formula to get revised values for a non-final layer
        Matrix derivativeOfActivationFunction = Matrix.add(activation, Matrix.multiplyByConstant(Matrix.multiply(activation, activation), -1));
        result = Matrix.multiply(Matrix.dotProduct(tansposedWeights, gb), derivativeOfActivationFunction);

        return result;
    }

    public static Matrix reviseValues(Matrix original, Matrix[] gradients, double eta, int m) {
        // use formula to get revised values
        Matrix sum = Train.summation(gradients);
        Matrix result = Matrix.add(original, Matrix.multiplyByConstant(sum, (eta/m)*-1));
        return result;
    }

    public static Matrix summation(Matrix[] matrices) {
        Matrix sum = matrices[0];

        // add up values of all matrices
        for (int i=1; i<matrices.length; i++) {
            sum = Matrix.add(sum, matrices[i]);
        }
        return sum;
    }

    public static Matrix[] getLayerValues(Matrix[][] values, int layer) {
        Matrix[] result = new Matrix[values.length];

        // get the n-th matrix from each matrix array and turn it into its own matrix
        for (int caseNum=0; caseNum<values.length; caseNum++) {
            result[caseNum] = values[caseNum][layer];
        } 

        return result;
    }
}

class Input {

    public static void writeValuesToFile(String fileName, Matrix[] values) throws IOException {
        double[][] elements;
        BufferedWriter outFile = new BufferedWriter(new FileWriter(fileName));

        for (Matrix currMatrix : values) {
            // get actual values from matrix
            elements = currMatrix.getElements();
            for (double[] row : elements) {
                // separate each element with a space
                for (double element : row) {
                    outFile.write(element + " ");
                }
                // separate each row with a ,
                outFile.write(", ");
            }
            // separate each matrix with a new line
            outFile.newLine();
        } 
        outFile.close();
    }

    public static Matrix[] getValuesFromFile(String fileName) throws IOException {
        BufferedReader csvReader = new BufferedReader(new FileReader(fileName));
        String currentLine;
        String[] rows;
        String[] elementString;
        double[][] elements;
        Matrix[] matrices = new Matrix[2];
        int count = 0;
        // iterate through every line (matrix)
        while ((currentLine = csvReader.readLine()) != null) {
            // separate rows by ,'s
            rows = currentLine.split(", ");
            
            // create double[][] to store elements
            elements = new double[rows.length][rows[0].trim().split(" ").length];
            
            // enter all element into the double[][]
            for (int i=0; i<rows.length; i++) {
                elementString = rows[i].trim().split(" ");
                for (int j=0; j<elementString.length; j++) {
                    elements[i][j] = Double.parseDouble(elementString[j]);
                }
            }
            // convert the double[][] into a matrix and record it
            matrices[count] = new Matrix(elements);
            count += 1;
        }
        csvReader.close();
        return matrices;
    }

    public static Matrix[][] getTestingData(String fileName) throws IOException{
        BufferedReader csvReader = new BufferedReader(new FileReader(fileName));
        String currentLine;
        int correctValue;
        ArrayList<Matrix> correctValues = new ArrayList<>();
        ArrayList<Matrix> pixels = new ArrayList<>();
        
        // iterate over all data entries
        while ((currentLine = csvReader.readLine()) != null) {
            // record first label value
            correctValue = Integer.parseInt(currentLine.substring(0, 1));
            correctValues.add(Matrix.transpose(Input.convertToOneHotVector(correctValue)));
            
            // record data as a matrix
            pixels.add(Matrix.transpose(Input.getNormalizedValues(currentLine)));
        }
        csvReader.close();

        // convert the arrayLists into matrix arrays
        Matrix[][] finalValues = {convertToMatrixArray(correctValues), convertToMatrixArray(pixels)};
        return finalValues;
    }

    public static Matrix[] convertToMatrixArray(ArrayList<Matrix> matrices) {
        Matrix matrixArray[] = new Matrix[matrices.size()];

        for (int i=0; i<matrices.size(); i++) {
            matrixArray[i] = matrices.get(i);
        }

        return matrixArray;
    }

    public static Matrix getNormalizedValues(String line) {
        double[] finalValues = new double[784];
        int currPixel = 0;
        int start = 2;
        
        // iterate over the entire string, recording the integers and normalizing them
        for (int i=2; i<line.length(); i++) {
            if (line.charAt(i) == ',') {
                finalValues[currPixel] = Integer.parseInt(line.substring(start, i)) / 255.0;
                currPixel += 1;
                start = i + 1;
            }
        }
        return new Matrix(finalValues);
    }

    public static Matrix convertToOneHotVector(int value) {
        double[] values = new double[10];
        // record the value as a 1 in value's position
        values[value] = 1;
        return new Matrix(values);
    }
}