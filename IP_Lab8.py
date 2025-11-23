import os
from PIL import Image
import numpy as np
import heapq
from collections import Counter
import json # Used for estimating size of metadata

# ==============================================================================
# Part A: Huffman Coding and Decoding
# ==============================================================================

# Node class for the Huffman tree
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Overload the less than operator for priority queue
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCompressor:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}

    def _make_frequency_dict(self, data):
        """Calculates the frequency of each pixel value."""
        return Counter(data)

    def _build_heap(self, frequency):
        """Builds a min-heap from the frequency dictionary."""
        heap = []
        for key, value in frequency.items():
            node = HuffmanNode(key, value)
            heapq.heappush(heap, node)
        return heap

    def _merge_nodes(self, heap):
        """Merges nodes to build the Huffman tree."""
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)
        return heap[0] # The root of the tree

    def _make_codes_helper(self, root, current_code):
        """Recursively builds the Huffman codes from the tree."""
        if root is None:
            return
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        self._make_codes_helper(root.left, current_code + "0")
        self._make_codes_helper(root.right, current_code + "1")

    def _generate_codes(self, root):
        """Generates the final Huffman codes."""
        self._make_codes_helper(root, "")

    def _get_encoded_text(self, data):
        """Encodes the image data using the generated Huffman codes."""
        encoded_text = ""
        for item in data:
            encoded_text += self.codes[item]
        return encoded_text

    def _pad_encoded_text(self, encoded_text):
        """Pads the encoded text to make its length a multiple of 8."""
        extra_padding = 8 - len(encoded_text) % 8
        if extra_padding == 8: # If already a multiple of 8
            extra_padding = 0
            
        padded_encoded_text = encoded_text + '0' * extra_padding
        padding_info = "{0:08b}".format(extra_padding)
        padded_encoded_text = padding_info + padded_encoded_text
        return padded_encoded_text

    def _get_byte_array(self, padded_encoded_text):
        """Converts the padded bit string to a byte array."""
        if len(padded_encoded_text) % 8 != 0:
            print("Error: Encoded text not padded correctly.")
            exit(0)
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

    def compress(self, image_path):
        """
        Compresses a color image file using Huffman coding by processing each channel.
        """
        print("--- Starting Huffman Compression ---")
        try:
            image = Image.open(image_path).convert('RGB') # Keep it as a color image
            image_data = np.array(image)
            shape = image_data.shape
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, None, None

        original_size = os.path.getsize(image_path)
        print(f"Original image size: {original_size / 1024:.2f} KB")

        compressed_channels = []
        reverse_maps = []
        total_compressed_size = 0

        for i in range(3): # For R, G, B channels
            print(f"  Compressing channel {i+1}/3...")
            # Reset state for each channel
            self.codes = {}
            self.reverse_mapping = {}

            flat_data = image_data[:,:,i].flatten()
            frequency = self._make_frequency_dict(flat_data)
            heap = self._build_heap(frequency)
            root = self._merge_nodes(heap)
            self._generate_codes(root)
            
            encoded_text = self._get_encoded_text(flat_data)
            padded_encoded_text = self._pad_encoded_text(encoded_text)
            byte_array = self._get_byte_array(padded_encoded_text)

            compressed_channels.append(byte_array)
            reverse_maps.append(self.reverse_mapping)
            total_compressed_size += len(byte_array)

        print(f"Total compressed size: {total_compressed_size / 1024:.2f} KB")
        if total_compressed_size > 0:
            compression_ratio = original_size / total_compressed_size
            print(f"Compression ratio: {compression_ratio:.2f}")

        print("--- Huffman Compression Finished ---")
        return compressed_channels, shape, reverse_maps

    def _remove_padding(self, padded_encoded_text):
        """Removes padding from the bit string."""
        padding_info = padded_encoded_text[:8]
        extra_padding = int(padding_info, 2)
        padded_encoded_text = padded_encoded_text[8:]
        
        if extra_padding == 0:
            return padded_encoded_text
            
        encoded_text = padded_encoded_text[:-extra_padding]
        return encoded_text

    def _decode_text(self, encoded_text, reverse_mapping):
        """Decodes the bit string back to pixel values."""
        current_code = ""
        decoded_pixels = []
        for bit in encoded_text:
            current_code += bit
            if current_code in reverse_mapping:
                character = reverse_mapping[current_code]
                decoded_pixels.append(character)
                current_code = ""
        return np.array(decoded_pixels)
    
    def decompress(self, compressed_channels, shape, reverse_maps):
        """
        Decompresses the data for each color channel and reconstructs the image.
        """
        print("\n--- Starting Huffman Decompression ---")
        
        decompressed_channels_data = []
        for i in range(3): # For R, G, B channels
            print(f"  Decompressing channel {i+1}/3...")
            compressed_data = compressed_channels[i]
            reverse_mapping = reverse_maps[i]
            
            bit_string = ""
            for byte in compressed_data:
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                
            encoded_text = self._remove_padding(bit_string)
            decoded_pixels = self._decode_text(encoded_text, reverse_mapping)
            decompressed_channels_data.append(decoded_pixels)

        # Reconstruct the image from the decompressed channels
        if not all(len(d) == shape[0] * shape[1] for d in decompressed_channels_data):
            print("Error: Decompressed data size does not match original image dimensions.")
            return None
            
        r = decompressed_channels_data[0].reshape(shape[0], shape[1])
        g = decompressed_channels_data[1].reshape(shape[0], shape[1])
        b = decompressed_channels_data[2].reshape(shape[0], shape[1])
        
        decompressed_image_array = np.stack((r, g, b), axis=-1)
        decompressed_image = Image.fromarray(decompressed_image_array.astype(np.uint8), 'RGB')
        
        print("--- Huffman Decompression Finished ---")
        return decompressed_image

# ==============================================================================
# Part B: Arithmetic Coding and Decoding (Chunked Implementation)
# ==============================================================================

class ArithmeticCompressor:
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size

    def _calculate_probability_ranges(self, data_chunk):
        """Generates a probability range table for symbols in a chunk."""
        chunk_len = len(data_chunk)
        frequencies = Counter(data_chunk)
        prob_ranges = {}
        low = 0.0
        # Sort for consistent ordering
        for symbol in sorted(frequencies.keys()):
            prob = frequencies[symbol] / chunk_len
            high = low + prob
            prob_ranges[symbol] = (low, high)
            low = high
        return prob_ranges

    def _encode_chunk(self, data_chunk, prob_ranges):
        """Encodes a single chunk of data."""
        low, high = 0.0, 1.0
        for symbol in data_chunk:
            sym_low, sym_high = prob_ranges[symbol]
            current_range = high - low
            high = low + current_range * sym_high
            low = low + current_range * sym_low
        return (low + high) / 2

    def _decode_chunk(self, encoded_value, prob_ranges, chunk_len):
        """Decodes a single chunk of data."""
        decoded_chunk = []
        low, high = 0.0, 1.0
        
        # Create a sorted list of symbols for consistent lookup
        sorted_symbols = sorted(prob_ranges.keys())
        
        for _ in range(chunk_len):
            current_range = high - low
            
            # Check for division by zero due to floating point precision limits
            if current_range == 0.0:
                # Cannot decode further, break the loop for this chunk
                break
                
            scaled_value = (encoded_value - low) / current_range

            found_symbol = None
            for symbol in sorted_symbols:
                sym_low, sym_high = prob_ranges[symbol]
                if sym_low <= scaled_value < sym_high:
                    found_symbol = symbol
                    decoded_chunk.append(found_symbol)
                    high = low + current_range * sym_high
                    low = low + current_range * sym_low
                    break
            
            if found_symbol is None:
                # Fallback for floating point precision issues
                decoded_chunk.append(sorted_symbols[-1])

        return decoded_chunk

    def compress(self, image_path):
        """
        Compresses an image by breaking each channel into chunks and encoding them.
        """
        print("--- Starting Arithmetic Compression (Chunked) ---")
        try:
            image = Image.open(image_path).convert('RGB')
            image_data = np.array(image)
            shape = image_data.shape
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, None

        original_size = os.path.getsize(image_path)
        print(f"Original image size: {original_size / 1024:.2f} KB")
        
        all_channels_data = []
        total_compressed_size = 0

        for i in range(3): # For R, G, B channels
            print(f"  Compressing channel {i+1}/3...")
            flat_data = image_data[:,:,i].flatten()
            
            compressed_chunks = []
            for j in range(0, len(flat_data), self.chunk_size):
                chunk = flat_data[j:j+self.chunk_size]
                if not len(chunk) > 0: continue

                prob_ranges = self._calculate_probability_ranges(chunk)
                encoded_val = self._encode_chunk(chunk, prob_ranges)
                
                # Store the encoded value, prob table, and original length for each chunk
                compressed_chunks.append((encoded_val, prob_ranges, len(chunk)))

            all_channels_data.append(compressed_chunks)
            
            # Estimate size: 8 bytes per float + size of probability tables
            # Convert numpy types to standard python types for JSON serialization
            channel_size = sum(8 + len(json.dumps({str(k): v for k, v in p.items()}).encode('utf-8')) for _, p, _ in compressed_chunks)
            total_compressed_size += channel_size


        print("--- Arithmetic Compression Finished ---")
        return all_channels_data, shape

    def decompress(self, all_channels_data, shape):
        """
        Decompresses image data by decoding each chunk in each channel.
        """
        print("\n--- Starting Arithmetic Decompression (Chunked) ---")
        
        reconstructed_channels = []
        for i, channel_chunks in enumerate(all_channels_data):
            print(f"  Decompressing channel {i+1}/3...")
            
            full_channel_data = []
            for encoded_val, prob_ranges, chunk_len in channel_chunks:
                decoded_chunk = self._decode_chunk(encoded_val, prob_ranges, chunk_len)
                
                # Handle cases where decoding stops early due to precision loss
                if len(decoded_chunk) < chunk_len:
                    # Get the last decoded pixel, or default to 0 if the chunk is empty
                    padding_value = decoded_chunk[-1] if decoded_chunk else 0
                    # Calculate how many pixels are missing
                    missing_pixels = chunk_len - len(decoded_chunk)
                    # Extend the chunk with the padding value to maintain correct length
                    decoded_chunk.extend([padding_value] * missing_pixels)

                full_channel_data.extend(decoded_chunk)
            reconstructed_channels.append(np.array(full_channel_data))

        if not all(len(d) == shape[0] * shape[1] for d in reconstructed_channels):
            print("Error: Decompressed data size does not match original image dimensions.")
            return None

        r = reconstructed_channels[0].reshape(shape[0], shape[1])
        g = reconstructed_channels[1].reshape(shape[0], shape[1])
        b = reconstructed_channels[2].reshape(shape[0], shape[1])
        
        decompressed_image_array = np.stack((r, g, b), axis=-1)
        decompressed_image = Image.fromarray(decompressed_image_array.astype(np.uint8), 'RGB')

        print("--- Arithmetic Decompression Finished ---")
        return decompressed_image

# ==============================================================================
# Main execution block
# ==============================================================================

if __name__ == '__main__':
    image_path = "sample_image.jpg"

    # --- Demonstrate Huffman Coding ---
    print("\n" + "="*25)
    print("DEMONSTRATING HUFFMAN CODING (COLOR)")
    print("="*25)
    huffman_compressor = HuffmanCompressor()
    compressed_channels, shape, reverse_maps = huffman_compressor.compress(image_path)
    
    if compressed_channels:
        decompressed_image_huffman = huffman_compressor.decompress(compressed_channels, shape, reverse_maps)
        if decompressed_image_huffman:
            output_path = "huffman_decompressed_color.jpg"
            print(f"\nSaving Huffman decompressed image to {output_path}...")
            decompressed_image_huffman.save(output_path)
            print("Image saved.")
    
    print("\n\n" + "="*25)
    print("DEMONSTRATING ARITHMETIC CODING (COLOR)")
    print("="*25)
    # --- Demonstrate Arithmetic Coding ---
    arithmetic_compressor = ArithmeticCompressor(chunk_size=15) 
    
    compression_results, img_shape = arithmetic_compressor.compress(image_path)
    
    if compression_results is not None:
        decompressed_img_arithmetic = arithmetic_compressor.decompress(compression_results, img_shape)
        
        if decompressed_img_arithmetic:
            output_path = "arithmetic_decompressed_color.jpg"
            print(f"\nSaving Arithmetic decompressed image to {output_path}...")
            decompressed_img_arithmetic.save(output_path)
            print("Image saved.")

