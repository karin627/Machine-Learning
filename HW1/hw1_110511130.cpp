# include<iostream>
# include<fstream>
# include<string>
# include<sstream>
# include<math.h>
# include<cmath>
# include<limits.h>
# include<vector>
using namespace std;
/*
data[n][11]:target
data[n][0]~data[n][10]:input

data[0][n]~data[9999][n]:training data
data[10000][n]~data[15817]:testing data
*/

const int M = 5;
const int KEY = 2; // have to minus one
const float LAMBDA = 0;

void calculateMean(ofstream& output, double mean[11]){
	for(int i=0;i<11;i++)
		mean[i] /= 10000.;
}

void calculateStandartDeviation(ofstream& output, double standart_deviation[11], double mean[11], double data[15817][12]){
	for(int i=0;i<11;i++){
		for(int j=0;j<10000;j++){
			standart_deviation[i] += (data[j][i]-mean[i])*(data[j][i]-mean[i]);
		}
		standart_deviation[i] /= 9999.;
		standart_deviation[i] = sqrt(standart_deviation[i]);
	}
}

void normalization(ofstream& output, double standard_deviation[11], double mean[11], double data[15817][12]){
	for(int i=0;i<11;i++)
		for(int j=0;j<15817;j++)
			data[j][i] = (data[j][i]-mean[i])/standard_deviation[i];
}

void calculateBasisFunction(ofstream& output, vector<vector<double>>& basis_function, double data[15817][12], string type){
	if(type == "training"){
		for(int i=0;i<10000;i++){
			for(int j=0;j<11;j++){
				for(int k=0;k<M;k++){
					if(j==0 && k==0)
						basis_function.push_back({1});
					else if(k==0)
						basis_function[i].push_back(1);
					else{
						double mu = 3*(-M+1+2*(k-1)*((double)(M-1)/(M-2)))/M;
						basis_function[i].push_back(1/(1+exp(-(data[i][j]-mu)/0.1)));
					}
				}
			}
		}
	}
	else if(type == "testing"){
		for(int i=10000;i<15817;i++){
			for(int j=0;j<11;j++){
				for(int k=0;k<M;k++){
					if(j==0 && k==0)
						basis_function.push_back({1});
					else if(k==0)
						basis_function[i].push_back(1);
					else{
						double mu = 3*(-M+1+2*(k-1)*((double)(M-1)/(M-2)))/M;
						basis_function[i].push_back(1/(1+exp(-(data[i][j]-mu)/0.1)));
					}
				}
			}
		}
	}

	// /*----------output training basis function----------*/
	// string path = "./observation/training_basis_function_"+to_string(M)+".txt";
	// output.open(path);
	// output<<"training basis function for M="<<M<<endl;
	// for(int i=0;i<10000;i++){
	// 	for(int j=0;j<11*M;j++){
	// 		output<<basis_function[i][j]<<" ";
	// 	}
	// 	output<<endl;
	// }
	// output.close();
}

void inverseMatrix(vector<vector<double>>& temp_mtx, vector<vector<double>>& inv_temp){
	double temp;
	for(int i=0;i<11*M;i++){
		if(i==0)
			inv_temp.push_back({1});
		else
			inv_temp.push_back({0});
		for(int j=1;j<11*M;j++){
			if(i==j)
				inv_temp[i].push_back(1);
			else
				inv_temp[i].push_back(0);
		}
	}

	for(int i=0;i<11*M;i++){
		temp = 1/temp_mtx[i][i];
		if(temp_mtx[i][i] == 0)
			cout<<i<<endl;
		for(int j=0;j<11*M;j++){
			temp_mtx[i][j] *= temp;
			inv_temp[i][j] *= temp;
		}
		for(int j=0;j<11*M;j++){
			if(i!=j){
				temp = temp_mtx[j][i];
				for(int k=0;k<11*M;k++){
					temp_mtx[j][k] -= temp_mtx[i][k] * temp;
					inv_temp[j][k] -= inv_temp[i][k] * temp;
				}
			}
		}
	}

}

void calculatePrediction(ofstream& output, vector<vector<double>>& basis_function, double data[15817][12], vector<double>& prediction){
	// ΦTΦ
	vector<vector<double>> temp_mtx;
	for(int i=0;i<11*M;i++){
		temp_mtx.push_back({0});
		for(int j=0;j<11*M-1;j++){
			temp_mtx[i].push_back(0);
		}
	}
	for(int i=0;i<11*M;i++){
		for(int j=0;j<11*M;j++){
			for(int k=0;k<10000;k++){
				temp_mtx[j][i] += basis_function[k][j] * basis_function[k][i];
			}
		}
	}
	for(int i=0;i<11*M;i++){
		temp_mtx[i][i] += 0.001;
		temp_mtx[i][i] += LAMBDA;
	}
	
	// (ΦTΦ)^-1
	vector<vector<double>> inv_temp;
	inverseMatrix(temp_mtx, inv_temp);

	// ΦTt
	vector<double> temp_mtx2(M*11, 0.0);
	for(int i=0;i<11*M;i++){
		for(int j=0;j<10000;j++){
			temp_mtx2[i] += basis_function[j][i] * data[j][11];
		}
	}

	// (ΦTΦ)^-1*ΦTt
	vector<double> weight(M*11, 0.0);
	for(int i=0;i<M*11;i++){
		for(int j=0;j<M*11;j++){
			weight[i] += inv_temp[i][j] * temp_mtx2[j];
		}
	}

	//training prediction
	if(LAMBDA==0)
		// output.open("./results/training_y_"+to_string(M)+".txt");
		output.open("training_y_"+to_string(M)+".txt");
	else
		// output.open("./results/training_y_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
		output.open("./results/training_y_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
	for(int i=0;i<10000;i++){
		for(int j=0;j<11*M;j++){
			if(j==0)
				prediction[i] = basis_function[i][j] * weight[j];
			else
				prediction[i] += basis_function[i][j] * weight[j];
		}
		output<<prediction[i]<<endl;
	}
	output.close();
	//testing prediction
	calculateBasisFunction(output, basis_function, data, "testing");
	if(LAMBDA==0)
		// output.open("./results/testing_y_"+to_string(M)+".txt");
		output.open("testing_y_"+to_string(M)+".txt");
	else
		// output.open("./results/testing_y_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
		output.open("testing_y_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
	for(int i=10000;i<15817;i++){
		for(int j=0;j<M*11;j++){
			if(j==0)
				prediction[i] = basis_function[i][j] * weight[j];
			else
				prediction[i] += basis_function[i][j] * weight[j];
		}
		output<<prediction[i]<<endl;
	}
	output.close();
}

void calculateMSE(ofstream& output, vector<double>& prediction, double data[15817][12], double MSE[2]){
	// training
	for(int i=0;i<10000;i++){
		MSE[0] += (prediction[i] - data[i][11])*(prediction[i] - data[i][11]);
	}
	MSE[0] /= 10000;

	//testing
	for(int i=10000;i<15817;i++){
		MSE[1] += (prediction[i] - data[i][11])*(prediction[i] - data[i][11]);
	}
	MSE[1] /= 5817;

	// output
	if(LAMBDA==0)
		// output.open("./results/MSEerror_accuracy_"+to_string(M)+".txt");
		output.open("MSEerror_accuracy_"+to_string(M)+".txt");
	else
		// output.open("./results/MSEerror_accuracy_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
		output.open("MSEerror_accuracy_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt");
	output<<"lambda="<<LAMBDA<<endl<<"MSE"<<endl;
	output<<"training : "<<MSE[0]<<endl;
	output<<"testing : "<<MSE[1]<<endl;
	output.close();
}

void calculateMSEValidation(ofstream& output, vector<double>& prediction, double data[15817][12], double MSE[5]){
	int iter = 0;
	output.open("./results/validation_mse_"+to_string(M)+".txt");
	output<<"validation MSE"<<endl;
	for(iter=0;iter<5;iter++){
		int validation_start = 2000*iter;
		int validation_end = 2000*iter+2000;
		for(int i=validation_start;i<validation_end;i++){
			MSE[iter] += (prediction[i] - data[i][11])*(prediction[i] - data[i][11]);
		}
		MSE[iter] /= 2000;
		MSE[5] += MSE[iter];
		output<<endl<<"MSE "<<iter<<" : "<<MSE[iter]<<endl;
	}
	MSE[5] /= 5;
	output<<endl<<"Average MSE for M="<<M<<" is "<<MSE[5]<<endl;
	output.close();
}

void calculateAccuracy(ofstream& output, vector<double>& prediction, double data[15817][12], double accuracy[2]){
	//training
	for(int i=0;i<10000;i++){
		if(data[i][11]!=0){
			accuracy[0] += fabs((prediction[i]-data[i][11])/data[i][11]);
		}
		else if(data[i][11]==0){
			accuracy[0] += fabs(prediction[i]-data[i][11]);
		}
	}
	accuracy[0] = 1-accuracy[0]/10000;

	//testing
	for(int i=10000;i<15817;i++){
		if(data[i][11]!=0){
			accuracy[1] += fabs((prediction[i]-data[i][11])/data[i][11]);
		}
		else if(data[i][11]==0){
			accuracy[1] += fabs(prediction[i]-data[i][11]);
		}
	}
	accuracy[1] = 1-accuracy[1]/5817;

	//output
	fstream append;
	if(LAMBDA==0)
		// append.open("./results/MSEerror_accuracy_"+to_string(M)+".txt", ios::app);
		append.open("MSEerror_accuracy_"+to_string(M)+".txt", ios::app);
	else
		// append.open("./results/MSEerror_accuracy_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt", ios::app);
		append.open("./results/MSEerror_accuracy_"+to_string(M)+"_lambda"+to_string(LAMBDA)+".txt", ios::app);
	append<<"accuracy"<<endl;
	append<<"training : "<<accuracy[0]<<endl;
	append<<"testing : "<<accuracy[1]<<endl;
	append.close();
}

void fiveFoldValidation(ofstream& output, vector<vector<double>>& basis_function, double data[15817][12], vector<double>& prediction){
	int iter = 0;
	double MSE[6] = {0.0};
	int validation_start;
	int validation_end;
	for(iter = 0;iter<5;iter++){
		validation_start = 2000*iter;
		validation_end = 2000*iter+2000;

		// ΦTΦ
		vector<vector<double>> temp_mtx;
		for(int i=0;i<11*M;i++){
			temp_mtx.push_back({0});
			for(int j=0;j<11*M-1;j++){
				temp_mtx[i].push_back(0);
			}
		}
		for(int i=0;i<11*M;i++){
			for(int j=0;j<11*M;j++){
				for(int k=0;k<validation_start;k++){
					temp_mtx[j][i] += basis_function[k][j] * basis_function[k][i];
				}
				for(int k=validation_end;k<10000;k++){
					temp_mtx[j][i] += basis_function[k][j] * basis_function[k][i];
				}
			}
		}

		for(int i=0;i<11*M;i++){
			temp_mtx[i][i] += 0.001;
		}

		// (ΦTΦ)^-1
		vector<vector<double>> inv_temp;
		inverseMatrix(temp_mtx, inv_temp);

		// ΦTt
		vector<double> temp_mtx2(M*11, 0.0);
		for(int i=0;i<11*M;i++){
			for(int j=0;j<validation_start;j++){
				temp_mtx2[i] += basis_function[j][i] * data[j][11];
			}
			for(int j=validation_end;j<10000;j++){
				temp_mtx2[i] += basis_function[j][i] * data[j][11];
			}
		}

		// (ΦTΦ)^-1*ΦTt
		vector<double> weight(M*11, 0.0);
		for(int i=0;i<11*M;i++){
			for(int j=0;j<11*M;j++){
				weight[i] += inv_temp[i][j] * temp_mtx2[j];
			}
		}

		//validation prediction
		for(int i=validation_start;i<validation_end;i++){
			for(int j=0;j<11*M;j++){
				if(j==0)
					prediction[i] = basis_function[i][j] * weight[j];
				else
					prediction[i] += basis_function[i][j] * weight[j];
			}
		}
	}

	//mse error
	calculateMSEValidation(output, prediction, data, MSE);
}

int main(){
	static double data[15817][12];
	double mean[11] = {0.0};
	double standart_deviation[11] = {0.0};
	double MSE[2]={0.0};
	double accuracy[2] = {0.0};
	vector<vector<double>> basis_function;
	vector<double> prediction(15817, 0);
	int i=0;
	int j=0;
	ifstream input;
	// ofstream output_test;
	ofstream output;

	/*------------input data-------------*/
	input.open("HW1.csv");
	if(input.is_open()){
		string line;
		// index
		getline(input, line);
		// data
		while(getline(input, line, ',')){
			string str_temp;
			stringstream ss(line);
			while(ss>>str_temp){
				// target
				if(j==0)
					data[i][11] = stof(str_temp);
				// input data
				else{
					data[i][j-1] = stof(str_temp);
					if(i<10000)
						mean[j-1] += data[i][j-1];
				}
				j++;
				if(j==12){
					i++;
					j=0;
				}
			}
		}
	}
	else{
		cout<<"Cannot open file!"<<endl;
	}
	input.close();

	/*----------output training x----------*/
	// output.open("./results/training_x_"+to_string(KEY+1)+".txt");
	output.open("training_x_"+to_string(KEY+1)+".txt");
	for(i=0;i<10000;i++){
		output<<data[i][KEY]<<endl;
	}
	output.close();

	/*----------output testing x----------*/
	// output.open("./results/testing_x_"+to_string(KEY+1)+".txt");
	output.open("testing_x_"+to_string(KEY+1)+".txt");
	for(i=10000;i<15817;i++)
		output<<data[i][KEY]<<endl;
	output.close();

	/*----------output training t----------*/
	// output.open("./results/training_t.txt");
	output.open("training_t.txt");
	for(int i=0;i<10000;i++){
		output<<data[i][11]<<endl;
	}
	output.close();

	/*----------output testing t----------*/
	// output.open("./results/testing_t.txt");
	output.open("testing_t.txt");
	for(int i=10000;i<15817;i++){
		output<<data[i][11]<<endl;
	}
	output.close();

	/*------------calculate mean and standard deviation-------------*/
	calculateMean(output, mean);
	calculateStandartDeviation(output, standart_deviation, mean, data);

	// /*------------do normalization-------------*/
	normalization(output, standart_deviation, mean, data);

	// /*------------training-------------*/
	calculateBasisFunction(output, basis_function, data, "training");
	// fiveFoldValidation(output, basis_function, data, prediction);
	calculatePrediction(output, basis_function, data, prediction);
	// /*------------calculate error------------*/
	calculateMSE(output, prediction, data, MSE);
	calculateAccuracy(output, prediction, data, accuracy);

	return 0;
}