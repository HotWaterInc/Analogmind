import torch
import math
import torch
import time
import torch
import torchvision.transforms as transforms


def load_everything():
    global storage, direction_network
    storage = StorageSuperset2()
    grid_dataset = 5

    storage.load_raw_data_from_others(f"data{grid_dataset}x{grid_dataset}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_dataset}x{grid_dataset}_connections.json")


def build_pil_image_from_path(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_preprocess = transform(image)

    return image_preprocess


def visualize_image_from_tensor(tensor_data, save_name: str):
    try:
        # Ensure the input is a numpy array
        if not isinstance(tensor_data, np.ndarray):
            tensor_data = np.array(tensor_data)

        # Check if the shape is correct

        # Transpose the array to (height, width, channels)
        img_data = np.transpose(tensor_data, (1, 2, 0))

        # Normalize the data if it's not already in 0-255 range
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)

        # Create image from array
        img = Image.fromarray(img_data, mode='RGB')

        # Display basic information about the image
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")

        # Save the image to a file
        output_path = f"./{save_name}.jpg"
        img.save(output_path, 'JPEG')
        print(f"Image saved successfully to {output_path}")

        return img

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def webots_radians_to_normal(x: float) -> float:
    if x < 0:
        x += 2 * math.pi
    return x


def test_images_accuracy():
    load_everything()
    width = 3
    height = 3
    grid_size = 5
    total_rotations = 24
    i, j = 0, 0
    rotation = 0
    x, y = get_position(width, height, grid_size, i, j, 0, 0.5)
    angle = get_angle(total_rotations, rotation)
    time.sleep(0.25)
    detach_robot_teleport_absolute(x, y)
    yield
    time.sleep(0.25)
    detach_robot_rotate_absolute(angle)
    yield
    detach_robot_sample_image_inference()
    yield

    global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()
    buffer = global_data_buffer.buffer

    image_data = buffer["data"]
    empty_global_data_buffer()

    input_batch_path = build_pil_image_from_path("data/dataset5x5/image1.jpeg")

    emb1 = pil_tensor_to_resnet18_embedding(input_batch_path)
    emb2 = process_webots_image_to_embedding(image_data)
    emb1 = squeeze_out_resnet_output(emb1)
    emb2 = squeeze_out_resnet_output(emb2)

    print(torch.dist(emb1, emb2))
    embedding_json_00 = storage.node_get_datapoints_tensor("0_0")[0].to(device)
    print(torch.dist(emb1, embedding_json_00))
    print(torch.dist(emb2, embedding_json_00))
    yield


def build_resnet18_embeddings_CORRECT():
    json_data_ids = read_other_data_from_file("data5x5_rotated24_image_id.json")
    max_id = json_data_ids[-1]["data"][-1][0]
    print(f"Max id: {max_id}")

    json_index = 0
    j_index = 0

    batch_size = 100
    image_array = []

    image_path = "data/dataset5x5"
    for i in range(0, max_id + 1):
        full_path = image_path + f"/image{i + 1}.jpeg"
        image = build_pil_image_from_path(full_path)
        embedding = pil_tensor_to_resnet18_embedding(image)
        embedding = squeeze_out_resnet_output(embedding)

        data_index = i % 24
        json_index = i // 24
        json_data_ids[json_index]["data"][data_index] = embedding.tolist()

        if i % batch_size == 0:
            print(f"Batch {i // batch_size} processed")

    write_other_data_to_file("data5x5_rotated24_image_embeddings.json", json_data_ids)
