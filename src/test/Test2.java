package test;
/*
 * This test is for bipolar representation
 */
import bplearning.BPLearning;

public class Test2 {
	public static void main(String[] args) {
		int epoch_times = 0;
		int converge_times = 0;
		for(int k =0 ; k< 1000 ; k++) {
		BPLearning BP = new BPLearning(2, 4, 0.2, 0, -1, 1);
		
		    double x[][] = {
			             	{-1,-1},
			             	{-1,+1},
			             	{+1,-1},
			             	{+1,+1}
				           };
                 double y[] = {-1,1,1,-1};
			
			for(int i = 0; i < 1000; i++) {
				double forEachStep = 0;
				//System.out.println("\n");
				for(int j = 0; j < 4; j++) {
					forEachStep = forEachStep + BP.train(x[j], y[j]);
				}
				double Error = 0.5 * forEachStep;
				//System.out.println(k + " Epoch "+i + " ERROR " +Error);
				System.out.println(k+1 +" "+ i +" "+ Error);
				if(Error <= 0.05) {
					//System.out.println("ERROR- " +i);
					epoch_times = epoch_times + i;
					converge_times++;
					break;
				}
			}
		}
		System.out.println(converge_times);
		System.out.println("Average - "+ epoch_times/converge_times);
	}
}
