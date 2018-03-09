package bplearning;
/*
 * Realize backpropagation learning 
 */

import java.util.Arrays;
import java.util.Random;
import java.io.File;
import java.io.IOException;

import interfaces.NeuralNetInterface;

public class BPLearning implements NeuralNetInterface{
	private int argNumInputs = 0;
	private int argNumHidden = 0;
	private double argMomentumTerm;
	private double[][] inputToHiddenNeuronWeights;
	private double[][] inputToHiddenNeuronPreviousWeights;
    private double[][] swapWeightInputToHidden;
    private double[] swapWeightHiddenToOutput;
	private double[] hiddenNeuronToOutputWeights;
	private double[] hiddenNeuronToOutputPreviousWeights;
	private double[] hiddenNeuralNetOutput;
	private double[] initialHidden;
	private int argA = 0;
	private int argB = 0;
	//private boolean activationType; // 1 - bipolar ; 0 - binary
	private double argRate;


	public BPLearning(int argNumInputs, int argNumHidden, double argRate,double argMomentumTerm, 
			 int argA, int argB) {
		this.argNumInputs = argNumInputs;
		this.argNumHidden = argNumHidden;
		this.argRate = argRate;
		this.argMomentumTerm = argMomentumTerm;
		//this.activationType = activationType;
		this.argA = argA;
		this.argB = argB;
		
		inputToHiddenNeuronWeights = new double[argNumHidden][argNumInputs+1];
		inputToHiddenNeuronPreviousWeights = new double[argNumHidden][argNumInputs+1];
		hiddenNeuralNetOutput = new double[argNumHidden];
		hiddenNeuronToOutputWeights = new double[argNumHidden+1];
		hiddenNeuronToOutputPreviousWeights = new double[argNumHidden+1];
	    swapWeightInputToHidden = new double[argNumHidden][argNumInputs+1];
	    swapWeightHiddenToOutput = new double[argNumHidden+1];
	    initialHidden = new double[argNumHidden];
		initializeWeights();
		zeroWeights();
	}
	
	//Step1: Initialize weights
	@Override
	public void initializeWeights() {
		Random random_value = new Random();
		for(int i=0; i<argNumHidden; i++) {	
			for(int j=0; j <argNumInputs; j++) {			
				inputToHiddenNeuronWeights[i][j] = random_value.nextDouble() - 0.5;
			}
	     	inputToHiddenNeuronWeights[i][argNumInputs] = random_value.nextDouble () - 0.5;
		}
		
		for(int i=0; i<argNumHidden; i++) {	
			hiddenNeuronToOutputWeights[i] = random_value.nextDouble() - 0.5;
	        }
		    hiddenNeuronToOutputWeights[argNumHidden] = random_value.nextDouble () - 0.5;
	}

	//Step2a. Perform a forward propagation
	@Override
	public double outputFor(double[] X) {
		Arrays.fill(initialHidden, 0);
		double output = 0;

		for(int i=0; i<argNumHidden; i++) {
			initialHidden[i] += inputToHiddenNeuronWeights[i][argNumInputs]*1.0;
			for(int j=0; j < argNumInputs; j++) {																			
				initialHidden[i] += inputToHiddenNeuronWeights[i][j] * X[j];
			}
			hiddenNeuralNetOutput[i] = customSigmoid(initialHidden[i]);
		}
		
		for(int i=0; i<argNumHidden; i++) {
			output +=  ( hiddenNeuronToOutputWeights[i] * hiddenNeuralNetOutput[i] );
		}
		output += hiddenNeuronToOutputWeights[argNumHidden]*1.0;
		return customSigmoid(output);
	}
	
	@Override
	public double train(double[] X, double argValue) {
		
		double y = outputFor(X);
		double outputsDerivative;
		
		/*For this assignment, we can compute the derivative directly
		 * 
		 * if(activationType) {
			outputsDerivative = 0.5 * (1 - Math.pow(y, 2));
		}else {
			outputsDerivative = y * (1 - y);
		}*/
		
		//Compute derivative of the general Sigmoid function
		outputsDerivative = (y - argA) * (argB - y)/(argB -argA);
		
		//Step2b. Back Propagation
		//get the output error signal
		double OutputErrorSignal = (argValue - y) * outputsDerivative;
	
	
        //Step2b. Get the error signal for hidden layers
      	double hiddenErrorSignal[] = new double[argNumHidden];
      		
      		/*
      		 * For this assignment, we can compute the derivative directly
      		 * if(activationType) {
      		for(int i = 0; i < argNumHidden; i++) {
      			hiddenErrorSignal[i] = 0.5 * ( 1 - Math.pow(hiddenNeuralNetOutput[i],2)) *
      					hiddenNeuronToOutputWeights[i] * OutputErrorSignal;
      		        }
      			}else {
      				for(int i = 0; i < argNumHidden; i++) {
      					hiddenErrorSignal[i] = hiddenNeuronToOutputWeights[i] * OutputErrorSignal *
      							hiddenNeuralNetOutput[i] * ( 1 - hiddenNeuralNetOutput[i]);
      				}
      			}*/
      
      	for(int i = 0; i < argNumHidden; i++) {
      			hiddenErrorSignal[i] = (hiddenNeuralNetOutput[i] - argA) * 
      					(argB - hiddenNeuralNetOutput[i])/(argB -argA) *
      					hiddenNeuronToOutputWeights[i] * OutputErrorSignal;
      		        }
      	
      //Step2c. Updating the weights for output to hidden layer neurons 
        System.arraycopy(hiddenNeuronToOutputWeights, 0, swapWeightHiddenToOutput, 
        		0, hiddenNeuronToOutputWeights.length);
		for(int i = 0; i < argNumHidden; i++) {			

				hiddenNeuronToOutputWeights[i] +=
						(calDeltaWeightHiddenToOutput(i) * argMomentumTerm)+
						(argRate * OutputErrorSignal * hiddenNeuralNetOutput[i]);
		}
		hiddenNeuronToOutputWeights[argNumHidden] += 
				(calDeltaWeightHiddenToOutput(argNumHidden) * argMomentumTerm)+
				(argRate * OutputErrorSignal * 1);
        System.arraycopy(swapWeightHiddenToOutput,
        		0, hiddenNeuronToOutputPreviousWeights, 
        		0, swapWeightHiddenToOutput.length);
      	
		//Step2c. Updating the weights for hidden to input layer weights
        for (int i = 0; i < argNumHidden; i++){
            System.arraycopy( inputToHiddenNeuronWeights[i], 0, 
            		swapWeightInputToHidden[i], 0, inputToHiddenNeuronWeights[i].length );
        }
        
		for(int i = 0; i < argNumHidden; i++) {
			for(int j = 0; j < argNumInputs; j++) {
	
					inputToHiddenNeuronWeights[i][j] += 
							(calDeltaWeightInputToHidden(i, j) *argMomentumTerm) + 
							(argRate * hiddenErrorSignal[i] * X[j]);
			}			
			        inputToHiddenNeuronWeights[i][argNumInputs] += 
			        		(calDeltaWeightInputToHidden(i, argNumInputs) *argMomentumTerm) + 
			        		(argRate * hiddenErrorSignal[i] * 1);
		}
		
		for (int i = 0; i < argNumHidden; i++){
	            System.arraycopy( swapWeightInputToHidden[i], 0, 
	            		inputToHiddenNeuronPreviousWeights[i], 0, 
	            		swapWeightInputToHidden[i].length );
	        }
	        
		return Math.pow((y - argValue), 2);
	}


	public double sigmoid(double x) {	
		/*if(activationType)
			return (1-Math.exp(-x)) / (1+Math.exp(-x));
		else
			return 1/(1 + Math.exp(-x));*/
		return 0;
	}
	
	@Override
	public double customSigmoid(double x) {
		return (argB - argA)/(1 + Math.exp(-x)) + argA;
	}
	
	@Override
	public void save(File argFile) {
		
	}

	@Override
	public void load(String argFileName) throws IOException {
		
	}
	
	@Override
	public double calDeltaWeightInputToHidden(int i, int j) {
        if (inputToHiddenNeuronPreviousWeights[i][j] != 0){
            return inputToHiddenNeuronWeights[i][j] - inputToHiddenNeuronPreviousWeights[i][j];
        } else{
            return 0;
        }
    }
	
	@Override
	public double calDeltaWeightHiddenToOutput(int i){
		 if (hiddenNeuronToOutputPreviousWeights[i] != 0){
	            return hiddenNeuronToOutputWeights[i] - hiddenNeuronToOutputPreviousWeights[i];
	        } else {
	            return 0;
	        }
	}
	
	@Override
	public void zeroWeights() {
		
		for(int i=0; i<argNumHidden; i++) 	
			for(int j=0; j <=argNumInputs; j++) {			
				inputToHiddenNeuronPreviousWeights[i][j] = 0;
			}
			
		for(int i=0; i<=argNumHidden; i++) {	
			hiddenNeuronToOutputPreviousWeights[i] = 0;
	        }	
	}
}
