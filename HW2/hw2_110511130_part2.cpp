# include<iostream>
# include<fstream>
# include<sstream>
# include<vector>
# include<cmath>
using namespace std;

const int FEATURE = 2;
const int CLASS = 3;
const int M = 3;
const int EPOCH = 1;
const double WEIGHT_INIT = 0.001;
const int TRAIN_TOTAL = 1300;
const int TEST_TOTAL = 750;
vector<int> TRAIN_NUM = {250, 350, 700};
vector<int> TEST_NUM = {300, 150, 300};

void innerProduct(vector<vector<double>>& answer, vector<vector<double>> v1, vector<vector<double>> v2, int i, int k, int j){
	for(int l=0;l<i;l++){
		for(int m=0;m<j;m++){
			for(int n=0;n<k;n++)
				answer[l][m] += v1[l][n] * v2[n][m];
		}
	}
}

void innerProductTranspose(vector<vector<double>>& answer, vector<vector<double>> v1, vector<vector<double>> v2, int i, int k, int j){
	for(int l=0;l<i;l++){
		for(int m=0;m<j;m++){
			for(int n=0;n<k;n++)
				answer[l][m] += v1[n][l] * v2[n][m];
		}
	}
}

void initialize2DVector(vector<vector<double>>& v, int i, int j, double value=0.0){
	for(int l=0;l<i;l++){
		vector<double> temp(j, value);
		v.push_back(temp);
	}
}

void inputData(vector<vector<double>>& x, vector<double>& t, vector<vector<double>>& mean, string filename){
	int i=0;
	int j=0;
	ifstream input;
	input.open(filename);
	if(input.is_open()){
		string line;
		// index
		getline(input, line);
		// data
		while(getline(input, line, ',')){
			string str_temp;
			stringstream ss(line);
			while(ss>>str_temp){
				if(i==0){
					t[j] = stof(str_temp);
					if(t[j]==0)
						t[j] = 3;
				}
				else{
					x[i-1][j] = stof(str_temp);
					if(filename == "HW2_training.csv")
						mean[i-1][int(t[j])-1] += stof(str_temp);
				}
				i++;
				if(i==FEATURE+1){
					j++;
					i=0;
				}
			}
		}
	}
	else{
		cout<<"Cannot open file!"<<endl;
	}
	input.close();
}

void outputNum(double n, string filename){
	ofstream output;
	output.open("./output_part2/"+filename);
	output<<n<<endl;
	output.close();
}

void output1DVector(vector<double>& v, int i, string filename){
	ofstream output;
	output.open("./output_part2/"+filename);
	for(int l=0;l<i;l++){
		output<<v[l]<<endl;
	}
	output.close();
}

void output2DVector(vector<vector<double>>& v, int i, int j, string filename){
	ofstream output;
	output.open("./output_part2/"+filename);
	for(int l=0;l<i;l++){
		for(int m=0;m<j;m++){
			output<<v[l][m]<<" ";
		}
		output<<endl;
	}
	output.close();
}

void inverseMatrix(vector<vector<double>>& temp_mtx, vector<vector<double>>& inv_temp){
	double temp;
	int N = temp_mtx[0].size();
	for(int i=0;i<N;i++){
		if(i==0)
			inv_temp.push_back({1});
		else
			inv_temp.push_back({0});
		for(int j=1;j<N;j++){
			if(i==j)
				inv_temp[i].push_back(1);
			else
				inv_temp[i].push_back(0);
		}
	}
	
	for(int i=0;i<N;i++){
		temp = 1/temp_mtx[i][i];
		if(temp_mtx[i][i] == 0)
			cout<<i<<endl;
		for(int j=0;j<N;j++){
			temp_mtx[i][j] *= temp;
			inv_temp[i][j] *= temp;
		}
		for(int j=0;j<N;j++){
			if(i!=j){
				temp = temp_mtx[j][i];
				for(int k=0;k<N;k++){
					temp_mtx[j][k] -= temp_mtx[i][k] * temp;
					inv_temp[j][k] -= inv_temp[i][k] * temp;
				}
			}
		}
	}
}

void calculateMean(vector<vector<double>>& mean){
	for(int i=0;i<FEATURE;i++){
		for(int j=0;j<CLASS;j++){
			mean[i][CLASS] += mean[i][j];
		}
		mean[i][CLASS] /= TRAIN_TOTAL;
	}
	for(int i=0;i<2;i++){
		mean[i][0] /= TRAIN_NUM[0];
		mean[i][1] /= TRAIN_NUM[1];
		mean[i][2] /= TRAIN_NUM[2];
	}
}

void calculateCovariance(vector<vector<double>>& covariance, vector<vector<double>> training_x, vector<vector<double>> mean){
	for(int i=0;i<FEATURE;i++){
		for(int j=0;j<FEATURE;j++){
			for(int k=0;k<TRAIN_TOTAL;k++){
				covariance[i][j] += (training_x[i][k]-mean[i][CLASS]) * (training_x[j][k]-mean[j][CLASS]);
			}
			covariance[i][j] /= TRAIN_TOTAL;
		}
	}
}

void calculateConfusion(vector<vector<double>>& confusion, vector<double> prediction, vector<double> target_data, int input_num){
	for(int i=0;i<input_num;i++){
		confusion[target_data[i]-1][prediction[i]-1]++;
	}
}

double calculateAccuracy(vector<vector<double>> confusion){
	double accuracy_denom = 0;
	double accuracy_nom = 0;
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<CLASS;j++){
			accuracy_denom += confusion[i][j];
			if(i==j)
				accuracy_nom += confusion[i][j];
		}
	}
	return accuracy_nom/accuracy_denom;
}

void GenerativeModel(vector<vector<double>> covariance, vector<vector<double>> mean, vector<vector<double>> input_data, vector<double> target_data, string label){
	vector<vector<double>> weight;
	vector<double> bias(CLASS, 0);
	vector<vector<double>> inv_covariance;
	vector<vector<double>> temp_mat; // 4x2
	vector<double> prediction;
	vector<vector<double>> a;
	vector<vector<double>> confusion;
	int input_num = 0;
	double accuracy = 0;
	bool calc_acc = 0;
	/*------------initialization------------*/
	initialize2DVector(weight, FEATURE, CLASS);
	initialize2DVector(temp_mat, CLASS, FEATURE);
	initialize2DVector(confusion, CLASS, CLASS);

    // input data for plotting
	if(input_data.empty()){
		input_num = 101*101;
		for(int i=0;i<101;i++){
			for(int j=0;j<101;j++){
				if(i==0 && j==0){
					input_data.push_back({(double)i});
					input_data.push_back({(double)j});
				}
				else{
					input_data[0].push_back((double)i);
					input_data[1].push_back((double)j);
				}
			}
		}
	}
	else{
		input_num = input_data[0].size();
		calc_acc = 1;
	}
	initialize2DVector(a, CLASS, input_num);

	// output2DVector(input_data, FEATURE, input_num, label+"/input_data.txt");

	/*------------calculate weight------------*/
	inverseMatrix(covariance, inv_covariance);
	innerProduct(weight, inv_covariance, mean, FEATURE, FEATURE, CLASS);
	
	/*------------calculate bias------------*/
	// μT Σ-1
	innerProductTranspose(temp_mat, mean, inv_covariance, CLASS, FEATURE, FEATURE);

	// μT Σ-1 μ + lnp(Ck)
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<FEATURE;j++){
			bias[i] += temp_mat[i][j] * mean[j][i];
		}
		bias[i] *= -0.5;
		bias[i] += log((double)TRAIN_NUM[i]/TRAIN_TOTAL);
	}

	/*------------calculate a(x)------------*/
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<input_num;j++){
			for(int k=0;k<FEATURE;k++){
				a[i][j] += weight[k][i] * input_data[k][j];
			}
			a[i][j] += bias[i];
		}
	}

	/*------------prediction------------*/
	// exponential
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<input_num;j++)
			a[i][j] = exp(a[i][j]);
	}

	// sum
	for(int i=0;i<input_num;i++){
		for(int j=0;j<CLASS;j++){
			if(j==0)
				prediction.push_back(a[j][i]);
			else
				prediction[i] += a[j][i];
		}
	}

	// divide
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<input_num;j++)
			a[i][j] /= prediction[j];
	}
	output2DVector(a, CLASS, input_num, label+"/probability.txt");

	// prediction
	for(int i=0;i<input_num;i++){
		int max = 0;
		prediction[i] = 1;
		for(int j=0;j<CLASS;j++){
			if(a[j][i]>a[prediction[i]-1][i]){
				max = a[j][i];
				prediction[i] = j+1;
			}
		}
	}
	output1DVector(prediction, input_num, label+"/prediction.txt");

	/*------------calculate confusion matrix------------*/
	if(calc_acc){
		calculateConfusion(confusion, prediction, target_data, input_num);
		accuracy = calculateAccuracy(confusion);
		output2DVector(confusion, CLASS, CLASS, label+"/confusion_matrix.txt");
		outputNum(accuracy, label+"/accuracy.txt");
	}
}

class DiscriminativeModel{
	public:
		void initialization();
		void construct(vector<vector<double>> x, vector<double> t);
		void constructDensityMatrix(vector<vector<double>> x);
		void calculateProbability(int e);
		void transformTarget(vector<double> t);
		void predict(vector<vector<double>> in, vector<double> t, string l);
		
		vector<vector<double>> density;
		vector<vector<double>> density_predict;
		vector<vector<double>> weight;
		vector<vector<double>> probability;
		vector<vector<double>> probability_predict;
		vector<vector<double>> target_transformed;
		int epoch = EPOCH;
};

void DiscriminativeModel::initialization(){
	initialize2DVector(weight, M, CLASS, WEIGHT_INIT);
	initialize2DVector(probability, CLASS, TRAIN_TOTAL);
	initialize2DVector(target_transformed, CLASS, TRAIN_TOTAL);
}

void DiscriminativeModel::construct(vector<vector<double>> training_x, vector<double> training_t){
	
	vector<vector<double>> weight_new; // 3x4
	
	initialize2DVector(weight_new, M, CLASS);
	

	initialization();
	constructDensityMatrix(training_x);
	// output2DVector(density, M, TRAIN_TOTAL, "discriminative/density.txt");
	transformTarget(training_t);

	/*------------iteration-------------*/
	for (int i=0;i<epoch;i++){
		calculateProbability(i);
		// output2DVector(probability, CLASS, TRAIN_TOTAL, "discriminative/probability_i"+to_string(i)+".txt");
		for(int j=0;j<CLASS;j++){
			vector<vector<double>> temp_mat; // 3x100
			vector<vector<double>> temp_mat2; // 3x3
			vector<vector<double>> temp_mat3; // 3x1
			vector<vector<double>> temp_mat4; // 3x1
			// ΦT R // R : probability[j]
			initialize2DVector(temp_mat, M, TRAIN_TOTAL);
			for(int k=0;k<M;k++){
				for(int l=0;l<TRAIN_TOTAL;l++){                       
					temp_mat[k][l] = density[k][l] * (probability[j][l]*(1-probability[j][l]));
				}
			}
			// output2DVector(temp_mat, M, TRAIN_TOTAL, "discriminative/temp_mat_i"+to_string(i)+"c"+to_string(j)+".txt");
			// ΦT R Φ
			initialize2DVector(temp_mat2, M, M);
			for(int l=0;l<M;l++){
				for(int m=0;m<M;m++){
					for(int n=0;n<TRAIN_TOTAL;n++)
						temp_mat2[l][m] += temp_mat[l][n] * density[m][n];
				}
			}
			// output2DVector(temp_mat2, M, M, "discriminative/temp_mat2_i"+to_string(i)+"c"+to_string(j)+".txt");

			// (ΦT R Φ)-1
			vector<vector<double>> inv_temp_mat2; // 3x3
			inverseMatrix(temp_mat2, inv_temp_mat2);
			// output2DVector(inv_temp_mat2, M, M, "discriminative/inverse_mat_i"+to_string(i)+"c"+to_string(j)+".txt");
			
			// ΦT (y-t)
			initialize2DVector(temp_mat3, M, 1);
			for(int k=0;k<M;k++){
				for(int l=0;l<TRAIN_TOTAL;l++){
					temp_mat3[k][0] += density[k][l]*(probability[j][l]-target_transformed[j][l]);
				}
			}
			// output2DVector(temp_mat3, M, 1, "discriminative/temp_mat3_i"+to_string(i)+"c"+to_string(j)+".txt");

			// (ΦT R Φ)-1 ΦT (y-t)
			initialize2DVector(temp_mat4, M, 1);
			for(int k=0;k<M;k++){
				for(int l=0;l<M;l++){
					temp_mat4[k][0] += inv_temp_mat2[k][l]*temp_mat3[l][0];
				}
			}
			// output2DVector(temp_mat4, M, 1, "discriminative/temp_mat4_i"+to_string(i)+"c"+to_string(j)+".txt");
			// Wnew
			for(int k=0;k<M;k++){
				weight_new[k][j] = weight[k][j] - temp_mat4[k][0];
			}

		}
		// Wnew
		for(int j=0;j<M;j++){
			for(int k=0;k<CLASS;k++){
				weight[j][k] = weight_new[j][k];
			}
		}
		output2DVector(weight_new, M, CLASS, "discriminative/weight_new_i"+to_string(i)+".txt");
	}
}

void DiscriminativeModel::constructDensityMatrix(vector<vector<double>> input_data){
	int input_num = input_data[0].size();
	vector<double> temp(input_num, 1);
	density.push_back(temp);
	density.push_back(input_data[0]);
	density.push_back(input_data[1]);
}

void DiscriminativeModel::calculateProbability(int epoch){
	vector<vector<double>> a;
	vector<double> a_sum;
	
	initialize2DVector(a, CLASS, TRAIN_TOTAL, 0);
	innerProductTranspose(a, weight, density, CLASS, M, TRAIN_TOTAL);
	// output2DVector(a, CLASS, TRAIN_TOTAL, "discriminative/a_i"+to_string(epoch)+".txt");
	// exponential
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<TRAIN_TOTAL;j++)
			a[i][j] = exp(a[i][j]);
	}
	
	// sum
	for(int i=0;i<TRAIN_TOTAL;i++){
		for(int j=0;j<CLASS;j++){
			if(j==0)
				a_sum.push_back(a[j][i]);
			else
				a_sum[i] += a[j][i];
		}
	}

	// output2DVector(a, CLASS, TRAIN_TOTAL, "discriminative/exp_a_i"+to_string(epoch)+".txt");

	// output1DVector(a_sum, TRAIN_TOTAL, "discriminative/a_sum_i"+to_string(epoch)+".txt");

	// divide
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<TRAIN_TOTAL;j++)
			probability[i][j] = a[i][j] / a_sum[j];
	}
}

void DiscriminativeModel::transformTarget(vector<double> target){
	for(int i=0;i<TRAIN_TOTAL;i++){
		target_transformed[target[i]-1][i] = 1;
	}
	// output2DVector(target_transformed, CLASS, TRAIN_TOTAL, "discriminative/target_transformed.txt");
}

void DiscriminativeModel:: predict(vector<vector<double>> input_data, vector<double> target_data, string label){
	
	vector<vector<double>> a;
	vector<double> a_sum;
	vector<vector<double>> confusion;
	vector<double> prediction;
	int input_num = 0;
	double accuracy = 0;
	bool calc_acc = 0;

	/*------------initialization-------------*/
	initialize2DVector(confusion, CLASS, CLASS);
	// input data for plotting
	if(input_data.empty()){
		input_num = 101*101;
		for(int i=0;i<101;i++){
			for(int j=0;j<101;j++){
				if(i==0 && j==0){
					input_data.push_back({(double)i});
					input_data.push_back({(double)j});
				}
				else{
					input_data[0].push_back((double)i);
					input_data[1].push_back((double)j);
				}
			}
		}
	}
	else{
		input_num = input_data[0].size();
		calc_acc = 1;
	}

	for(int i=0;i<input_num;i++){
		prediction.push_back(0);
	}
	
	density.clear();
	constructDensityMatrix(input_data);
	initialize2DVector(a, CLASS, input_num, 0);
	initialize2DVector(probability_predict, CLASS, input_num);
	innerProductTranspose(a, weight, density, CLASS, M, input_num);

	output2DVector(a, CLASS, input_num, label+"/ak.txt");
	// exponential
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<input_num;j++)
			a[i][j] = exp(a[i][j]);
	}
	// sum
	for(int i=0;i<input_num;i++){
		for(int j=0;j<CLASS;j++){
			if(j==0)
				a_sum.push_back(a[j][i]);
			else
				a_sum[i] += a[j][i];
		}
	}
	// divide
	for(int i=0;i<CLASS;i++){
		for(int j=0;j<input_num;j++)
			probability_predict[i][j] = a[i][j] / a_sum[j];
	}

	//prediction
	for(int i=0;i<input_num;i++){
		int max = 0;
		prediction[i] = 1;
		for(int j=0;j<CLASS;j++){
			if(probability_predict[j][i]>probability_predict[prediction[i]-1][i]){
				max = probability_predict[j][i];
				prediction[i] = j+1;
			}
		}
	}

	output1DVector(prediction, input_num, label+"/prediction_e"+to_string(EPOCH)+".txt");
	if(calc_acc){
		calculateConfusion(confusion, prediction, target_data, input_num);
		accuracy = calculateAccuracy(confusion);
		output2DVector(confusion, CLASS, CLASS, label+"/confusion_matrix_e"+to_string(EPOCH)+".txt");
		outputNum(accuracy, label+"/accuracy_e"+to_string(EPOCH)+".txt");
	}

}

int main(){
	vector<vector<double>> training_x;
	vector<double> training_t(TRAIN_TOTAL, 0);
	vector<vector<double>> testing_x;
	vector<double> testing_t(TEST_TOTAL, 0);
	vector<vector<double>> mean;
	vector<vector<double>> covariance;
	ifstream input;
	int i=0;
	int j=0;

	/*------------initialization-------------*/
	initialize2DVector(training_x, FEATURE, TRAIN_TOTAL);
	initialize2DVector(testing_x, FEATURE, TEST_TOTAL);
	initialize2DVector(mean, FEATURE, CLASS+1);
	initialize2DVector(covariance, FEATURE, FEATURE);

	/*------------input data-------------*/
	inputData(training_x, training_t, mean, "HW2_training.csv");
	inputData(testing_x, testing_t, mean, "HW2_testing.csv");

	/*----------output data----------*/
	// output2DVector(training_x, FEATURE, TRAIN_TOTAL, "training_x.txt");
	// output2DVector(testing_x, FEATURE, TEST_TOTAL, "testing_x.txt");
	// output1DVector(training_t, TRAIN_TOTAL, "training_t.txt");
	// output1DVector(testing_t, TEST_TOTAL, "testing_t.txt");

	/*----------generative model----------*/
	vector<vector<double>> temp_in;
	vector<double> temp_in2;
	calculateMean(mean);
	calculateCovariance(covariance, training_x, mean);
	output2DVector(covariance, FEATURE, FEATURE, "generative/covariance.txt");
	GenerativeModel(covariance, mean, temp_in, temp_in2, "generative/plot");
	GenerativeModel(covariance, mean, training_x, training_t, "generative/training");
	GenerativeModel(covariance, mean, testing_x, testing_t, "generative/testing");

	/*----------discriminative model----------*/
	DiscriminativeModel discriminative_model;
	discriminative_model.construct(training_x, training_t);
	discriminative_model.predict(temp_in, temp_in2, "discriminative/plot");
	discriminative_model.predict(training_x, training_t, "discriminative/training");
	discriminative_model.predict(testing_x, testing_t, "discriminative/testing");

	return 0;
}