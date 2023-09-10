import AmeiseDataset as Ameise

frames = Ameise.unpack_frames("samples/test_file_01.4mse")
print(frames[-1].images[2].timestamp)
