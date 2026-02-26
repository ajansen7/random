import streamlit as st
# In a real scenario, you would import your API library here
# import openai 

def generate_dino_prompt(dino_type, accessories, setting, art_style):
    """
    Constructs a detailed prompt for the image generator based on user inputs.
    """
    # 1. Base Subject
    prompt = f"A {dino_type} dinosaur"
    
    # 2. Add Accessories (if any)
    if accessories:
        # Join list into a string like "wearing a top hat and sunglasses"
        items = ", ".join(accessories)
        prompt += f" wearing {items}"
    
    # 3. Setting the Scene (Environment)
    prompt += f", located in a {setting}"
    
    # 4. Style & fidelity boosters (Crucial for "Nano Banana" style results)
    prompt += f". {art_style}. Highly detailed, cinematic lighting, 8k resolution, masterpiece."
    
    return prompt

def mock_api_call(prompt):
    """
    This simulates sending the prompt to an API.
    Replace this with actual API code (e.g., client.images.generate(...))
    """
    # For this demo, we just return the prompt so you can see what is sent
    return f"https://via.placeholder.com/1024x1024.png?text=Imagine:+{prompt.replace(' ', '+')}"

# --- APP UI ---

st.set_page_config(page_title="Toddler Dino Creator", page_icon="🦖")

st.title("🦖 Toddler Dino Creator")
st.markdown("Build your own dinosaur and watch it come to life!")

# 1. The Dinosaur
st.sidebar.header("1. Choose your Dino")
dino_type = st.sidebar.selectbox(
    "Select a Base Dinosaur",
    ["T-Rex", "Triceratops", "Stegosaurus", "Velociraptor", "Brachiosaurus", "Pterodactyl"]
)

# 2. The Look
st.sidebar.header("2. Accessorize!")
accessories = st.sidebar.multiselect(
    "What is the dino wearing?",
    ["a colorful party hat", "cool sunglasses", "a superhero cape", "rain boots", "a bowtie", "a astronaut helmet", "a tutu"]
)

# 3. The Vibe
st.sidebar.header("3. The Setting")
setting_map = {
    "Realistic Jungle": "lush prehistoric fern jungle with sunlight filtering through trees",
    "Volcano": "rocky landscape with an active volcano in the background",
    "Space": "surface of the moon with earth in the background",
    "City": "modern city street crossing a crosswalk",
    "Ice Age": "snowy tundra with glaciers"
}
setting_choice = st.sidebar.selectbox("Where is your dino?", list(setting_map.keys()))

# 4. Art Style
st.sidebar.header("4. Art Style")
style_choice = st.sidebar.radio(
    "Pick a style:",
    ["Photorealistic (National Geographic style)", "3D Render (Pixar style)", "Cartoon (Hand drawn)"]
)

# --- GENERATION LOGIC ---

if st.button("Generate Dino!", type="primary"):
    with st.spinner("Hatching your dinosaur..."):
        # 1. Build the specific setting string based on the dino type if needed
        # (Here we use the map, but you could add logic like: if T-Rex, make jungle scarier)
        full_setting = setting_map[setting_choice]
        
        # 2. Construct the final prompt
        final_prompt = generate_dino_prompt(dino_type, accessories, full_setting, style_choice)
        
        # 3. Call the Image Generator
        # In a real app, this would return an image URL
        image_url = mock_api_call(final_prompt)
        
        # 4. Display Result
        st.success(f"Created a {dino_type}!")
        
        # Displaying the generated prompt for debugging/learning
        with st.expander("See the prompt we sent to the AI"):
            st.code(final_prompt)
            
        # Display the image
        st.image(image_url, caption=f"Your custom {dino_type}")