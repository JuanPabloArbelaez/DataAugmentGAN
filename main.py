from gan_models import *
from visualize import *
from tune import *



# Performance
best_p_real, best_gen_name = find_optimal()

def check_performance():
    accuracies = []
    p_real_all = torch.linspace(0, 1, 21)
    for p_real_vis in tqdm(p_real_all):
        accuracies += [eval_augmentation(p_real_vis, best_gen_name, n_test=4)]

    plt.plot(p_real_all.tolist(), accuracies)
    plt.ylabel("Accuracy")
    _ = plt.xlabel("Percent Real Images")


def view_generations():
    examples = [33, 66, 99, 11, 5]
    train_images = torch.load("insect_train.pt")["images"][examples]
    train_labels = torch.load("insect_train.pt")["labels"][examples]

    one_hot_labels = get_one_hot_labels(train_labels.to(device), n_classes).float()
    fake_noise = get_noise(len(train_images), z_dim, device=device)
    noise_and_labels = combine_vectors(fake_noise, one_hot_labels)

    gen = Generator(generator_input_dim).to(device)
    gen.load_state_dict(torch.load(best_gen_name))

    fake = gen(noise_and_labels)
    show_tensor_images(torch.cat([train_images.cpu(), fake.cpu()]))


if __name__ == '__main__':
    check_performance()
    # view_generations()
