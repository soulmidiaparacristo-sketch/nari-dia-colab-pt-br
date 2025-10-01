import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Use the function from inference.py
    model = Dia.from_pretrained("Soul25r/Diaptbr", compute_dtype="float16", device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    seed: Optional[float] = None,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            audio_data, sr = sf.read(audio_prompt_input)
            # Check if audio_data is valid
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(audio_data)  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation

        start_time = time.time()
# Apply fixed seed if provided (seguro contra None/NaN/valores inválidos)
try:
    if seed is not None and not np.isnan(seed):
        s = int(seed)
        torch.manual_seed(s)
        np.random.seed(s)
        print(f"Using fixed seed: {s}")
    else:
        print("No fixed seed provided, using random behavior.")
except Exception as e:
    print(f"Warning: failed to set seed ({seed}): {e}")


        
        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=False):  # Auto-patched for float32 on T4
                output_audio_np = model.generate(
                    text_input,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                    use_torch_compile=False,  # Keep False for Gradio stability
                    audio_prompt=prompt_path_for_generate,
                )

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor
            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}")

            # Explicitly convert to int16 to prevent Gradio warning
            if output_audio[1].dtype == np.float32 or output_audio[1].dtype == np.float64:
                audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                output_audio = (output_sr, audio_for_gradio)
                print("Converted audio to int16 for Gradio output.")

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}")
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}")

    return output_audio


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                type="filepath",
            )
            
            seed_input = gr.Number(label="Seed do Falante (deixe em branco para aleatório)", value=42, precision=0)
            
            with gr.Accordion("Parâmetros de Geração", open=False):
                gr.Markdown("""
**Expressões Sonoras Reconhecidas:**

(laughs) (risos), (clears throat) (pigarreia), (sighs) (suspiro), (gasps) (ofega), (coughs) (tosse),  
(singing) (cantando), (sings) (canta), (mumbles) (murmura), (beep) (bip), (groans) (geme),  
(sniffs) (fungada), (claps) (bate palmas), (screams) (grita), (inhales) (inspira), (exhales) (expira),  
(applause) (aplausos), (burps) (arrota), (humming) (cantarola), (sneezes) (espirra),  
(chuckle) (risadinha), (whistles) (assobia)
""")
                max_new_tokens = gr.Slider(
        label="Máximo de Tokens (Duração do Áudio)",
        minimum=860,
        maximum=3072,
        value=model.config.data.audio_length,
        step=50,
        info="Define o tempo máximo do áudio gerado. Quanto maior o valor, mais longo será o áudio.",
    )
                cfg_scale = gr.Slider(
        label="Escala CFG (Força de Adesão ao Texto)",
        minimum=1.0,
        maximum=5.0,
        value=3.0,
        step=0.1,
        info="Controla o quanto o modelo segue fielmente o texto. Valores maiores aumentam a precisão.",
    )
                temperature = gr.Slider(
        label="Temperatura (Aleatoriedade)",
        minimum=1.0,
        maximum=1.5,
        value=1.3,
        step=0.05,
        info="Define o nível de criatividade na fala. Valores baixos geram fala mais robótica; altos, mais variada.",
    )
                top_p = gr.Slider(
        label="Top P (Amostragem por Núcleo)",
        minimum=0.80,
        maximum=1.0,
        value=0.95,
        step=0.01,
        info="Filtra as palavras mais prováveis até atingir a probabilidade P. Afeta naturalidade e controle.",
    )
                cfg_filter_top_k = gr.Slider(
        label="Filtro Top K para CFG",
        minimum=15,
        maximum=50,
        value=30,
        step=1,
        info="Limita a escolha às K palavras mais prováveis durante a geração. Valores maiores dão mais liberdade.",
    )
                speed_factor_slider = gr.Slider(
        label="Fator de Velocidade",
        minimum=0.8,
        maximum=1.0,
        value=0.94,
        step=0.02,
        info="Ajusta a velocidade da fala. 1.0 é a velocidade normal; valores menores tornam a fala mais lenta.",
    )


            run_button = gr.Button("Gerar Áudio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Áudio Gerado",
                type="numpy",
                autoplay=False,
            )

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            seed_input,   # <-- ADICIONADO
        ],
        outputs=[audio_output],  # Add status_output here if using it
        api_name="generar_audio",
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "./example_prompt.mp3"  # Adjust if needed
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway! ",
            None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
        [
            "[S1] Você tem controle total sobre os roteiros e as vozes. Sou suspeito pra falar, mas acho que vencemos com folga. (laughs) Difícil discordar. Obrigado por ouvir esta demonstração. ",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
            ],
            outputs=[audio_output],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)
