#include <json.hpp>
#include <muParser.h>
#include "muparser_fun.hpp"
#include <iostream>
#include<vector>
#include<functional>
#include <cmath>
#include<fstream>

using json = nlohmann::json;

//PARAMETERS
//I define a struct to hold all the parameters that are required by the main function
//I put in some default values, which were the ones suggested in the exercise, since some fields would have not been default initialized and to speed up testing

//after having tested different parameters, one of the better values for alpha to ensure convergence for all methods is 0.1
struct Parameters{
    double er=1.0e-6;
    double es=1.0e-6;
    double sigma=0.3;
    std::function<double (double, double)> f=[] (double x,double y){return x*y+4*pow(x,4)+y*y+3*x;};
    std::function<double (double, double)> grad1=[] (double x,double y){return y+16*pow(x,3)+3;};
    std::function<double (double, double)> grad2=[] (double x,double y){return x+2*y;};
    
    double alpha=0.1;
    int maxiter=1000;
    std::vector<double> x0={0.0,0.0};
    double learning_rate=0.1;
};

//FUNCTION DEFINITIONS

//auxiliary functions

//vector operations
double norm(std::vector<double> const & vec);
std::vector<double> sub(std::vector<double> a, std::vector<double> b);
std::vector<double> sum(std::vector<double> a, std::vector<double> b);
std::vector<double> mul(std::vector<double> const & a, double c);

//different ways of selecting the learning rate
double Armijo(Parameters parameters,int k,double alpha,std::vector<double> x);
double exponential_decay(Parameters parameters,int k,double alpha,std::vector<double> x);
double inverse_decay(Parameters parameters,int k,double alpha,std::vector<double> x);

//Different strategies to minimize
std::vector<double> gradient_descent(Parameters const &parameters);
std::vector<double> Nesterov(Parameters parameters);
std::vector<double> momentum(Parameters parameters);

//here I define my main function
std::vector<double> minimize(Parameters const & parameters);


int main(){
    std::ifstream file("data.json");
    json data = json::parse(file);
    Parameters parameters;
    //I initialize the parameters struct from the json file
    std::string funString = data.value("f","");
    std::string dfunString1 = data.value("grad1","");
    std::string dfunString2=data.value("grad2","");

    parameters.f=MuparserFun(funString);
    parameters.grad1=MuparserFun(dfunString1);
    parameters.grad2=MuparserFun(dfunString2);
    parameters.er=data.value("er",1e-6);
    parameters.es=data.value("es",1e-6);
    parameters.alpha=data.value("alpha",0.1);
    parameters.learning_rate=data.value("learning_rate",0.1);
    parameters.maxiter=data.value("maxiter",1000);
    parameters.sigma=data.value("sigma",0.3);
    parameters.x0={data.value("x0_0",0.0),data.value("x0_1",0.0)}; //sistemarea 

    //Here I call the main function requested by the exercise
    auto x_min= minimize(parameters);
    //I print the results 
    std::cout<<x_min[0]<<","<<x_min[1]<<std::endl;
    std::cout<<"The value of the function is: "<<parameters.f(x_min[0],x_min[1])<<std::endl;
}
//euclidian norm
double norm(std::vector<double> const & vec){
    double sum=0;
    for(auto x:vec){
        sum+=x*x;
    }
    return sqrt(sum);
}
//I define sum and subrtraction between vectors and scalar-vector multiplication
std::vector<double> sub(std::vector<double> a, std::vector<double> b){
    if(a.size()!=b.size()) {
        std::cerr << "Dimensions are not compatible"<<std::endl;
        return std::vector<double>{};
    }
    else{
        std::vector<double> res;
        for(std::size_t i=0;i<a.size();++i) {
            res.push_back(a[i] - b[i]);
        }
        return res;
    }
}

std::vector<double> sum(std::vector<double> a, std::vector<double> b){
    if(a.size()!=b.size()) {
        std::cerr << "Dimensions are not compatible"<<std::endl;
        return std::vector<double>{};
    }
    else{
        std::vector<double> res;
        for(std::size_t i=0;i<a.size();++i) {
            res.push_back(a[i] + b[i]);
        }
        return res;
    }
}

std::vector<double> mul(std::vector<double> const & a,double c){
    std::vector<double> res;
    for(auto x: a){
        res.push_back(c*x);
    }
    return res;
}

//learning rate related functions
//Armijo method
double Armijo(Parameters parameters,int k,double alpha,std::vector<double> x){   
    if(parameters.sigma>0.5 || parameters.sigma<0){
        std::cerr<<"The value for sigma cannot be used, please input another value in the interval (0,0.5)"<<std::endl;
        return -1.0;
    }
    std::vector<double> grad={parameters.grad1(x[0],x[1]),parameters.grad2(x[0],x[1])};
    while(parameters.f(x[0],x[1])-parameters.f(sub(x,mul(grad,alpha))[0],sub(x,mul(grad,alpha))[1])<parameters.sigma*alpha*norm(grad)*norm(grad))
        alpha/=2;
    return alpha;
}

//exponential decay
double exponential_decay(Parameters parameters,int k,double alpha,std::vector<double> x) {
        return parameters.alpha*exp(-1*parameters.sigma*k);
}

//inverse decay
double inverse_decay(Parameters parameters,int k,double alpha,std::vector<double> x){  
    return parameters.alpha/(1+parameters.sigma*k);
}

//minimizing strategies
//At the beginning of each of the minimizing strategies functions I ask the user to choose a method to compute alpha and call back the same minimizing
//function if a wrong choice is input

//gradient descent
std::vector<double> gradient_descent(Parameters const & parameters){
    std::vector<double> x_old;
    std::vector<double> x=parameters.x0;
    double alpha=parameters.alpha;
    unsigned int choice;
    std::cout<<"What strategy would you like to follow?\nType 1 for Armijo,\nType 2 for exponential decay,\nType 3 for inverse decay"<<std::endl;
    std::cin>>choice;
    std::function<double (Parameters,int,double,std::vector<double>)> method;
    if(choice==1)
        method=Armijo;
    else if(choice==2)
        method=exponential_decay;
    else if(choice==3)
        method=inverse_decay;
    else {
        std::cerr << "You have not input one of the three allowed values, please compile again";
        return gradient_descent(parameters);
    }
    for(int k=0;k<parameters.maxiter;++k){
        x_old=x;
        alpha= method(parameters,k,alpha,x);
        std::vector<double> grad={parameters.grad1(x[0],x[1]),parameters.grad2(x[0],x[1])};
        x=sub(x,mul(grad,alpha));
        if(norm(sub(x,x_old))<parameters.es){
            break;
        }
        if(norm(grad)<parameters.er)
            break;
    }
    
    return x;
}

//Nesterov method
std::vector<double> Nesterov(Parameters parameters){
    std::vector<double> x_old=parameters.x0;
    std::vector<double> x=sub(x_old,mul({parameters.grad1(x_old[0],x_old[1]),parameters.grad2(x_old[0],x_old[1])},parameters.alpha));
    double alpha=parameters.alpha;
    unsigned int choice;
    std::cout<<"What strategy would you like to follow?\nType 1 for Armijo,\nType 2 for exponential decay,\nType 3 for inverse decay"<<std::endl;
    std::cin>>choice;
    std::function<double (Parameters,int,double,std::vector<double>)> method;
    if(choice==1)
        method=Armijo;
    else if(choice==2)
        method=exponential_decay;
    else if(choice==3)
        method=inverse_decay;
    else {
        std::cerr << "You have not input one of the three allowed values, please compile again";
        return Nesterov(parameters);
    }
    std::vector<double> y(x.size());
    double eta=parameters.learning_rate;
    for(int k=0;k<parameters.maxiter;++k){
        alpha= method(parameters,k,alpha,x);
        y=sum(x,mul(sub(x,x_old),eta));
        std::vector<double> grad={parameters.grad1(y[0],y[1]),parameters.grad2(y[0],y[1])};
        x=sub(x,mul(grad,alpha));
        x_old=x;
        if(norm(sub(x,x_old))<parameters.es)
            break;
        std::vector<double> gradx={parameters.grad1(x[0],x[1]),parameters.grad2(x[0],x[1])};
        if(norm(gradx)<parameters.er)
            break;
    }
    return x;
}

//gradient descent with momentum
std::vector<double> momentum(Parameters parameters){
    std::vector<double> x_old;
    std::vector<double> x=parameters.x0;
    double alpha=parameters.alpha;
    unsigned int choice;
    std::cout<<"What strategy would you like to follow?\nType 1 for exponential decay,\nType 2 for inverse decay"<<std::endl;
    std::cin>>choice;
    std::function<double (Parameters,int,double,std::vector<double>)> method;
    if(choice==1)
        method=exponential_decay;
    else if(choice==2)
        method=inverse_decay;
    else {
        std::cerr << "You have not input one of the three allowed values, please compile again";
        return momentum(parameters);
    }
    std::vector<double> grad0={parameters.grad1(x[0],x[1]),parameters.grad2(x[0],x[1])};
    auto d=mul(grad0,-1*parameters.alpha);
    double eta=parameters.learning_rate;
    for(int k=0;k<parameters.maxiter;++k){
        x_old=x;
        alpha= method(parameters,k,alpha,x);
        x=sum(x,d);
        std::vector<double> grad={parameters.grad1(x[0],x[1]),parameters.grad2(x[0],x[1])};
        d=sub(mul(d,eta),mul(grad,alpha));
        if(norm(sub(x,x_old))<parameters.es)
            break;
        if(norm(grad)<parameters.er)
            break;
    }
    return x;
}

//principal function
std::vector<double> minimize(Parameters const & parameters){
    int choice=-1;
    //I ask the user to choose their preferred minimizing strategy and initialize a variable that keeps track of the choice and use it to call one auxiliary function
    std::cout<<"Choose the preferred strategy for minimizing:\ntype 1 for gradient descent,\n2 for momentum\n or 3 for Nesterov"<<std::endl;
    std::cin>>choice;
    std::vector<double> x_min (parameters.x0.size());
    if(choice==1)
        x_min=gradient_descent(parameters);
    else if(choice==2)
        x_min=momentum(parameters); //da definire
    else if(choice==3)
        x_min=Nesterov(parameters);
    //If the value of choice does not correspond to anything I call this function again
    else{
        std::cerr<<"You have not input one of the three allowed values, please compile again"<<std::endl;
        return minimize(parameters);
        }
    return x_min;

}