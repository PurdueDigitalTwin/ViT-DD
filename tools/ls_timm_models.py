import timm

if __name__ == "__main__":
    print(timm.list_models('*deit3*', pretrained=True))
