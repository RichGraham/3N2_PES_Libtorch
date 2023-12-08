#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";


  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  //inputs.push_back(torch::ones(5));
  //inputs.push_back(torch::linspace(0, 1, 51));
  inputs.push_back(torch::tensor({{0.30402942999999998940907630640140268951654434204102, \
         0.24148845999999998812590717989223776385188102722168,\
         0.36185402999999999318347931875905487686395645141602,\
         0.26655042000000001012338657346845138818025588989258,\
         0.25468997999999998249620603019138798117637634277344,\
         0.21565207999999999599616273826541146263480186462402,\
         0.20971572999999998909714804540271870791912078857422,\
         0.18464316999999999535653216753416927531361579895020,\
         0.20889119999999999910400561020651366561651229858398,\
         0.19418668000000000040117242861015256494283676147461,\
         0.20697134000000000364494212590216193348169326782227,\
	 0.20038775999999999810619044637860497459769248962402}}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << '\n';
  //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
