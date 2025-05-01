# Technical Note for AAPM 2025 Grand Challenge

Yu Sun

yu.sun@petermac.org

Peter MacCallum Cancer Centre, Melbourne, Australia

# DockerHub ID
The Docker image is on DockerHub: 

# Run the inference

# Method
The model contains two sub-models.
1. The provided based model was modified as the first model. The last layer (meta information) was dropped and the input channel is 7. It's still using the MedNeXt v1.
   * Input channel: 7, the first 7 channel of the provided data loader.
   * Output channel: 1
   * Model type: B
3. The second model (CNN) revised the output by taking the output of the first, as well as meta information, beam plates, etc.
   * Input channel 4: The previous output (pred), along with body mask, ptv mask, meta information. 
       * The 4 channel input was processed by a CNN encoder -> out1
   * Beam plate was taken as an extra argument.
       * It's processed by another encoder -> out2
   * The results (pred, out1, out2) were added and passed to another CNN for final merge.
