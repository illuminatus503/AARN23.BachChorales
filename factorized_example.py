import torch
import tltorch


def factorized_numel(factorized_tensor: tltorch.FactorizedTensor):
    total_numel = 0
    total_factors = 0
    for factor in factorized_tensor.decomposition:
        if isinstance(factor, torch.Tensor):
            total_numel += factor.numel()
            total_factors += 1
        else:
            for factor_ in factor:
                total_numel += factor_.numel()
                total_factors += 1

    return total_numel, total_factors


def main():
    # Paso 1: Crear un Tensor de Ejemplo
    # Supongamos un tensor de dimensiones 10x10x10x10
    original_tensor = torch.randn(1000, 1000)

    # Paso 2: Descomponer el Tensor
    # Usaremos una descomposición CP con un rango especificado
    factorized_tensor: tltorch.FactorizedTensor = tltorch.FactorizedTensor.from_tensor(
        original_tensor, rank=0.01, factorization="tt"
    )

    # Paso 3: Comparar el Número de Parámetros
    # Calculamos el número de parámetros del tensor original y del factorizado
    num_params_original = original_tensor.numel()
    num_params_factorized, num_factors = factorized_numel(factorized_tensor)

    print(f"Número de parámetros original: {num_params_original}")
    print(
        f"Número de parámetros después de la factorización: {num_params_factorized}, con {num_factors} factores"
    )

    print("Close?", torch.allclose(original_tensor, factorized_tensor.to_tensor()))
    print(torch.norm(original_tensor - factorized_tensor.to_tensor(), p=torch.inf))


if __name__ == "__main__":
    main()
