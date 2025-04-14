use tch::{
    kind::Kind, // Keep Kind import
    // Import nn, ModuleT, Optimizer, OptimizerConfig
    nn::{self, ModuleT, Optimizer, OptimizerConfig}, // <--- Keep ModuleT here
    Device,
    IndexOp, // <--- Add IndexOp trait for .i() method
    Tensor,
};
use std::{
    convert::TryFrom, // Import TryFrom trait for f64::try_from
    fs::File,
    io::{self, Read}, // Group io imports
    path::{Path, PathBuf}, // Use PathBuf for owned paths
    error::Error, // Import the Error trait for Box<dyn Error>
    time::Duration, // Import Duration for timeout
};
use flate2::read::GzDecoder;
// Ensure reqwest is in Cargo.toml:
// reqwest = { version = "0.11", features = ["blocking", "rustls-tls"], default-features = false }
// Or:
// reqwest = { version = "0.11", features = ["blocking", "native-tls"], default-features = false }


// Define the structure of the neural network. This struct holds the layers
// and other parameters. It's essentially the blueprint for our model.
#[derive(Debug)]
struct Net {
    // Linear layer 1: 784 input features, 128 output features.
    fc1: nn::Linear,
    // Linear layer 2: 128 input features, 64 output features.
    fc2: nn::Linear,
    // Linear layer 3: 64 input features, 10 output features (for the 10 digits).
    fc3: nn::Linear,
}

// Implement the *trainable* neural network module trait (ModuleT).
// This provides the .train() and .eval() methods via default implementations.
impl nn::ModuleT for Net {
    // Define the forward pass for training/evaluation.
    // The `_train` argument indicates the mode, prefix with _ as it's unused here.
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor { // <-- Renamed train to _train
        // The `nn::Module` trait must be in scope to use `forward` on submodules like nn::Linear
        use tch::nn::Module; // <--- Import Module trait here

        // Flatten the input tensor. MNIST images are 28x28, so we reshape
        // them into a 784-element vector.
        let xs = xs.view([-1, 784]); // Use [-1, 784] for automatic batch size inference

        // Apply layers. Note: If using Dropout or BatchNorm, you'd use the `_train` flag here.
        let x = self.fc1.forward(&xs).relu();
        let x = self.fc2.forward(&x).relu();
        // Apply the third linear layer. No activation function here because
        // cross_entropy_for_logits expects raw logits.
        self.fc3.forward(&x)
    }

    // Note: We are relying on the default implementations of train() and eval()
    // provided by the ModuleT trait, even though calling them directly seems problematic.
}

// Function to create the neural network. This function initializes the layers
// with the specified dimensions.
fn net(vs: &nn::VarStore) -> Net {
    // Call vs.root() directly for each layer to avoid move errors.
    // vs.root() returns a new Path value each time.
    let fc1 = nn::linear(vs.root() / "fc1", 784, 128, Default::default());
    let fc2 = nn::linear(vs.root() / "fc2", 128, 64, Default::default());
    let fc3 = nn::linear(vs.root() / "fc3", 64, 10, Default::default());
    Net { fc1, fc2, fc3 }
}

// Function to load the MNIST dataset. This function downloads the dataset
// if it's not already present and returns the training and testing data.
fn load_mnist(
    train_images_path: &str,
    train_labels_path: &str,
    test_images_path: &str,
    test_labels_path: &str,
) -> Result<(Tensor, Tensor, Tensor, Tensor), Box<dyn Error>> { // Use Box<dyn Error>

    // Helper function to read gz files
    fn read_gz_file<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, io::Error> { // Use io::Error
        let file = File::open(path.as_ref())?; // Ensure path is used correctly
        let mut gz_decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        gz_decoder.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    // Helper function to read MNIST images
    fn read_images(data: &[u8]) -> Result<Tensor, Box<dyn Error>> {
        // Read header information safely
        if data.len() < 16 {
            return Err("Invalid image file header: too short".into());
        }
        // Use try_into() for safe slicing and conversion to fixed-size arrays
        let magic_number = u32::from_be_bytes(data[0..4].try_into()?);
        if magic_number != 2051 {
             return Err(format!("Invalid magic number for images: {}", magic_number).into());
        }
        let num_images = u32::from_be_bytes(data[4..8].try_into()?) as i64;
        let rows = u32::from_be_bytes(data[8..12].try_into()?) as i64;
        let cols = u32::from_be_bytes(data[12..16].try_into()?) as i64;
        // Check for potential overflow before calculating expected_len
        let num_pixels = rows.checked_mul(cols).ok_or("Image dimensions overflow")?;
        let data_size = num_images.checked_mul(num_pixels).ok_or("Total image data size overflow")?;
        let expected_len = 16usize.checked_add(data_size as usize).ok_or("Expected file length overflow")?;

        if data.len() != expected_len {
            return Err(format!("Invalid image file length: expected {}, got {}", expected_len, data.len()).into());
        }

        let image_data = &data[16..];
        // Create tensor and normalize
        let tensor = Tensor::from_slice(image_data)
            .view([num_images, rows * cols]) // Use array for shape
            .to_kind(Kind::Float) / 255.0;
        Ok(tensor)
    }

    // Helper function to read MNIST labels
    fn read_labels(data: &[u8]) -> Result<Tensor, Box<dyn Error>> {
         // Read header information safely
        if data.len() < 8 {
            return Err("Invalid label file header: too short".into());
        }
         // Use try_into() for safe slicing and conversion to fixed-size arrays
        let magic_number = u32::from_be_bytes(data[0..4].try_into()?);
         if magic_number != 2049 {
             return Err(format!("Invalid magic number for labels: {}", magic_number).into());
        }
        let num_labels = u32::from_be_bytes(data[4..8].try_into()?) as i64;
        let expected_len = 8usize.checked_add(num_labels as usize).ok_or("Expected label file length overflow")?;

         if data.len() != expected_len {
            return Err(format!("Invalid label file length: expected {}, got {}", expected_len, data.len()).into());
        }
        let label_data = &data[8..];
        // Create tensor
        let tensor = Tensor::from_slice(label_data).to_kind(Kind::Int64).view(num_labels);
        Ok(tensor)
    }

    // --- Download logic ---
    // Create data directory if it doesn't exist
    let data_dir = Path::new("data");
    if !data_dir.exists() {
        println!("Creating data directory: {}", data_dir.display());
        std::fs::create_dir(data_dir)?;
    }
    // Function to download a file
    fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn Error>> {
        // Check if file already exists and is not empty
        if path.exists() {
            match std::fs::metadata(path) {
                Ok(metadata) => {
                    if metadata.len() > 0 {
                        // println!("File {} already exists.", path.display()); // Less verbose
                        return Ok(());
                    } else {
                        println!("File {} exists but is empty. Re-downloading...", path.display());
                    }
                }
                Err(e) => {
                     // File might not exist or other permission error, proceed to download
                     println!("Warning: Could not get metadata for {}: {}. Attempting download.", path.display(), e);
                }
            }
        }


        println!("Downloading {} to {}...", url, path.display());
        // Ensure reqwest dependency is available
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300)) // <--- Increased timeout to 300 seconds (5 minutes)
            .build()?;
        let response = client.get(url).send()?; // Use client

        // Check if the request was successful
        let status = response.status();
        if !status.is_success() {
             // Consume the response body to close the connection cleanly, even on error.
             let body_text = response.text().unwrap_or_else(|e| format!("Could not read response body: {}", e));
             return Err(format!("Failed to download {}: HTTP {} - {}", url, status, body_text).into());
        }
        // Create file and copy content
        let mut file = File::create(path)?;
        let content = response.bytes()?; // Read bytes first
        io::copy(&mut content.as_ref(), &mut file)?; // Copy from byte slice using io::copy
        Ok(())
    }
    // Define paths using PathBuf for ownership
    let train_images_pb: PathBuf = train_images_path.into();
    let train_labels_pb: PathBuf = train_labels_path.into();
    let test_images_pb: PathBuf = test_images_path.into();
    let test_labels_pb: PathBuf = test_labels_path.into();

    // --- Use HTTPS URLs for MNIST download ---
    download_file("https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", &train_images_pb)?;
    download_file("https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", &train_labels_pb)?;
    download_file("https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", &test_images_pb)?;
    download_file("https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", &test_labels_pb)?;
    // --- End Download Logic ---


    // Load training images
    println!("Loading training images from {}...", train_images_path);
    let train_images_data = read_gz_file(train_images_path)?;
    let train_images = read_images(&train_images_data)?;

    // Load training labels
    println!("Loading training labels from {}...", train_labels_path);
    let train_labels_data = read_gz_file(train_labels_path)?;
    let train_labels = read_labels(&train_labels_data)?;

    // Load test images
    println!("Loading test images from {}...", test_images_path);
    let test_images_data = read_gz_file(test_images_path)?;
    let test_images = read_images(&test_images_data)?;

    // Load test labels
    println!("Loading test labels from {}...", test_labels_path);
    let test_labels_data = read_gz_file(test_labels_path)?;
    let test_labels = read_labels(&test_labels_data)?;

    println!("Dataset loaded successfully.");
    Ok((train_images, train_labels, test_images, test_labels))
}


// Function to calculate the accuracy of the model. This function compares the
// model's predictions to the true labels and returns the percentage of
// correct predictions.
// Takes ModuleT now to potentially use train=false if needed, though not strictly necessary here.
fn accuracy<M: nn::ModuleT>(net: &M, images: &Tensor, targets: &Tensor, device: Device, batch_size: i64) -> Result<f64, Box<dyn Error>> { // Use Box<dyn Error>
    if images.size().is_empty() || targets.size().is_empty() {
        return Ok(0.0); // Handle empty tensors
    }
    let num_images = images.size()[0];
    if num_images == 0 { return Ok(0.0); }

    let mut total_correct: i64 = 0;
    let num_batches = (num_images + batch_size - 1) / batch_size;

    // Ensure model is in eval mode for accuracy calculation
    // net.eval(); // Caller should handle setting eval mode via no_grad block

    for i in 0..num_batches {
        let start = i * batch_size;
        let current_batch_size = (num_images - start).min(batch_size);
        if current_batch_size <= 0 { continue; }

        let batch_images = images.i((start..start + current_batch_size, ..)).to_device(device);
        let batch_labels = targets.i(start..start + current_batch_size).to_device(device);

        let logits = net.forward_t(&batch_images, false); // Use forward_t with train=false
        let predicted = logits.argmax(1, false);

        if predicted.size() == batch_labels.size() {
            // .sum() returns a scalar tensor; .int64_value(&[]) extracts the i64 value directly.
            total_correct += predicted.eq_tensor(&batch_labels).sum(Kind::Int64).int64_value(&[]); // <-- Fix: Removed .iter().sum()
        } else {
            eprintln!(
               "Warning: Mismatched shapes in accuracy batch. Logits shape: {:?}, Predicted shape: {:?}, Labels shape: {:?}",
               logits.size(), predicted.size(), batch_labels.size()
            );
        }
    }

    Ok((total_correct as f64 / num_images as f64) * 100.0)
}


fn main() -> Result<(), Box<dyn Error>> { // Use Box<dyn Error>
    // Import ModuleT trait into main scope - needed for train/eval methods to resolve
    // use tch::nn::ModuleT; // <--- Keep ModuleT import here (or at top level)

    // Set the device to either CUDA (GPU) if available, or the CPU.
    let device = Device::cuda_if_available(); // Simpler way to get default device
    println!("Using device: {:?}", device);

    // Define dataset paths relative to the "data" directory
    let data_dir = Path::new("data");
    // Ensure data directory exists before creating paths within it
     if !data_dir.exists() {
        println!("Creating data directory: {}", data_dir.display());
        std::fs::create_dir(data_dir)?;
    }

    let train_images_path = data_dir.join("train-images-idx3-ubyte.gz").to_str().ok_or("Invalid train images path")?.to_string();
    let train_labels_path = data_dir.join("train-labels-idx1-ubyte.gz").to_str().ok_or("Invalid train labels path")?.to_string();
    let test_images_path = data_dir.join("t10k-images-idx3-ubyte.gz").to_str().ok_or("Invalid test images path")?.to_string();
    let test_labels_path = data_dir.join("t10k-labels-idx1-ubyte.gz").to_str().ok_or("Invalid test labels path")?.to_string();


    // Load the MNIST dataset (handles download internally now)
    // Pass paths as &str
    let (train_images, train_labels, test_images, test_labels) =
        load_mnist(&train_images_path, &train_labels_path, &test_images_path, &test_labels_path)?;

    // Data is initially on CPU, move it to the target device later in batches or all at once if memory allows.

    // Create a new variable store. This is where the model's parameters (weights
    // and biases) are stored. The VarStore device determines where parameters live.
    let vs = nn::VarStore::new(device);
    // Define the neural network. Don't need mut anymore as train/eval calls are removed.
    let net = net(&vs); // <--- Removed mut

    // Create an optimizer using the builder pattern (updated API).
    let learning_rate = 1e-2; // Define learning rate
    // Need to import Sgd struct specifically if using it like this
    use tch::nn::Sgd; // <--- Import Sgd struct
    let mut opt: Optimizer = Sgd { // Explicitly type opt as Optimizer
        momentum: 0.9, // Set momentum
        ..Default::default() // Use defaults for other parameters
    }
    .build(&vs, learning_rate)?; // Use variable for learning rate

    // Set the batch size. This is the number of images we process at a time.
    let batch_size: i64 = 64; // Explicitly type as i64
    let test_batch_size: i64 = 512; // Batch size for evaluation
    // Set the number of epochs. An epoch is one complete pass through the
    // training data.
    let num_epochs = 10;
    // Calculate the number of batches per epoch.
    let num_train_images = train_images.size()[0]; // size() returns Vec<i64>
    // Ensure num_train_images is not zero before division
    if num_train_images == 0 {
        return Err("Training dataset is empty.".into());
    }
    // Use ceiling division for num_batches to include the last partial batch
    let num_batches = (num_train_images + batch_size - 1) / batch_size;

     if num_batches == 0 && num_train_images > 0 {
        println!(
            "Warning: Calculated zero batches. Num train images: {}, Batch size: {}. Check logic.",
             num_train_images, batch_size
        );
    }


    println!(
        "Starting training: {} epochs, Batch size: {}, Batches per epoch: {}, Learning rate: {}",
        num_epochs, batch_size, num_batches, learning_rate
    );

    // Training loop.
    for epoch in 0..num_epochs {
        let mut running_loss = 0.0;
        let mut num_correct_in_epoch: i64 = 0; // Use i64 for count
        let mut total_processed_in_epoch: i64 = 0;

        // Set model to training mode using standard method call
        // WORKAROUND: Removing call due to persistent E0599 error.
        // net.train();

        // Iterate over the training data in batches.
        for batch_index in 0..num_batches {
            // Get the current batch of images and labels.
            let start = batch_index * batch_size;
            let current_batch_size = (num_train_images - start).min(batch_size); // Calculate actual batch size
             if current_batch_size <= 0 { continue; } // Skip if batch size is zero or negative

            // Move data batch to the target device
            let batch_images = train_images.i((start..start + current_batch_size, ..)).to_device(device);
            let batch_labels = train_labels.i(start..start + current_batch_size).to_device(device);


            // Forward pass: compute the model's predictions (logits).
            // Use forward_t with train=true (even though train() call is removed)
            let logits = net.forward_t(&batch_images, true);

            // Compute the loss using cross_entropy_for_logits (built-in).
            let loss = logits.cross_entropy_for_logits(&batch_labels);

            // Backpropagation: compute the gradients and update weights.
            opt.zero_grad(); // Zero gradients before backward pass
            loss.backward();
            opt.step(); // Update weights

            // Accumulate loss for reporting - use f64::try_from
            let loss_value = f64::try_from(&loss).unwrap_or(f64::NAN); // Handle potential errors gracefully
            if !loss_value.is_nan() {
                running_loss += loss_value * current_batch_size as f64; // Weight loss by batch size
            }


             // Calculate accuracy for this batch and accumulate correct count
            let predicted_classes = logits.argmax(1, false);
            // Ensure predicted_classes and batch_labels have compatible shapes/types
            if predicted_classes.size() == batch_labels.size() {
                 // .sum() returns a scalar tensor; .int64_value(&[]) extracts the i64 value directly.
                 let correct_count: i64 = predicted_classes.eq_tensor(&batch_labels).sum(Kind::Int64).int64_value(&[]); // <-- Fix: Removed .iter().sum()
                 num_correct_in_epoch += correct_count;
            } else {
                 eprintln!("Warning: Mismatched shapes in training batch accuracy. Logits shape: {:?}, Labels shape: {:?}", logits.size(), batch_labels.size());
            }
            total_processed_in_epoch += current_batch_size;


            // Optional: Print batch loss less frequently
            if (batch_index + 1) % 100 == 0 || batch_index == num_batches - 1 { // Print every 100 batches and the last batch
                 println!(
                    "Epoch: {}/{}, Batch: {}/{}, Batch Loss: {:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_index + 1,
                    num_batches,
                    loss_value,
                );
            }
        } // End of batches for epoch

        // Calculate average loss and accuracy for the epoch based on total processed items
        let epoch_loss = if total_processed_in_epoch > 0 { running_loss / total_processed_in_epoch as f64 } else { 0.0 };
        let epoch_accuracy = if total_processed_in_epoch > 0 {
             (num_correct_in_epoch as f64 / total_processed_in_epoch as f64) * 100.0
        } else {
             0.0
        };


        // Evaluate the model on the test set at the end of each epoch.
        // Set model to evaluation mode using standard method call
        // WORKAROUND: Removing call due to persistent E0599 error.
        // net.eval();

        // Add explicit return type annotation to the closure
        let (test_accuracy, test_loss) = tch::no_grad(|| -> Result<(f64, f64), Box<dyn Error>> {
            // Calculate test loss (similar to accuracy calculation)
             let num_test_images = test_images.size()[0];
             if num_test_images == 0 { return Ok((0.0, 0.0)); } // Handle empty test set

             let mut total_test_loss = 0.0;
             let num_test_batches = (num_test_images + test_batch_size - 1) / test_batch_size;

             for i in 0..num_test_batches {
                 let start = i * test_batch_size;
                 let current_batch_size = (num_test_images - start).min(test_batch_size);
                 if current_batch_size <= 0 { continue; }

                 let batch_images = test_images.i((start..start + current_batch_size, ..)).to_device(device);
                 let batch_labels = test_labels.i(start..start + current_batch_size).to_device(device);

                 // Use forward_t with train=false (even though eval() call is removed)
                 let test_logits = net.forward_t(&batch_images, false);
                 let loss = test_logits.cross_entropy_for_logits(&batch_labels);

                 let current_loss_value = f64::try_from(&loss).unwrap_or(f64::NAN);
                  if !current_loss_value.is_nan() {
                     total_test_loss += current_loss_value * current_batch_size as f64;
                  }
             }
             let avg_test_loss = if num_test_images > 0 { total_test_loss / num_test_images as f64 } else { 0.0 };

            // Calculate test accuracy using the accuracy function
            let avg_test_acc = accuracy(&net, &test_images, &test_labels, device, test_batch_size)?; // Pass necessary args

            Ok((avg_test_acc, avg_test_loss)) // Ensure final Ok matches closure signature
        })?; // Propagate potential errors from accuracy function


        println!(
            "Epoch: {}/{}, Train Loss: {:.4}, Train Acc: {:.2}%, Test Loss: {:.4}, Test Acc: {:.2}%",
            epoch + 1,
            num_epochs,
            epoch_loss,
            epoch_accuracy,
            test_loss, // This is now average test loss
            test_accuracy // This is now average test accuracy
        );
    } // End of epochs

    println!("Training complete.");

    // Optional: Save the trained model weights
    let model_path = data_dir.join("mnist_model.ot");
    vs.save(&model_path)?; // Save in the data directory using PathBuf
    println!("Model weights saved to {}", model_path.display());

    Ok(())
}
