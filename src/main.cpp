#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
// Todo: Learn About Armadillo
#include <armadillo>
// Todo: Learn Matplot++
#include <matplot/matplot.h>

#include "DenseLayer.hpp"

struct GeneratedData{
    arma::mat X;
    arma::vec y;
};

GeneratedData ReadData(){
    arma::mat X(300,2,arma::fill::zeros);
    arma::vec y(100,arma::fill::zeros);
    std::string lineData;
    // As The Exe is in Cmake-build-debug
    // This needs to change in different env
    std::fstream data("../Data/data.txt",std::ios::in);
    bool working_X = true;
    int countD_X=0;
    while(std::getline(data,lineData)){
        if(lineData == "Start_X:"){
           working_X = true;
        }else if(lineData == "Start_Y:"){
            working_X = false;
        }else if(working_X == true){
            lineData.erase(0,3);
            lineData.erase(lineData.size()-1);
            auto it = lineData.find(" ");
            if(it != std::string::npos){
                if(countD_X > 299){
                    std::cout << lineData << "\n";
                    throw "Unexpected Data";
                }
                double numx = std::stod(lineData.substr(0,it));
                double numy = std::stod(lineData.substr(it,lineData.size()-1));
                X(countD_X,0) = numx;
                X(countD_X,1) = numy;
                countD_X++;
            }
            std::cout << X << std::endl;
        }else {
            std::cout << lineData << std::endl;
        }
    }
    GeneratedData ret;
    ret.X = X;
    ret.y = y;
    return ret;
}


int main() {
    arma::mat random(2,5,arma::fill::randu);
    arma::mat zeros(2,5,arma::fill::zeros);
    DenseLayer dl(2,5);
    std::cout << dl.getBiases() << std::endl;
    std::cout << dl.getWeight() << std::endl;
    GeneratedData G = ReadData();
    return EXIT_SUCCESS;
}