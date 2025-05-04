#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
class MainStart:
    def __init__(self):
        self.models = []  # List to store models
        self.current_model = None  # Currently selected model

    def main_menu(self):
        while True:
            print("Main Menu:")
            print("1. Create New Model")
            print("2. Add Model To Simulation")
            print("3. Archive Model")
            print("4. Start Simulation")
            print("6. End Simulation")
            print("7. Check Current Results")
            print("8. Exit")

            choice = input("Enter your choice: ")

            if choice == '1':
                self.createNewModel()
            elif choice == '2':
                self.add_model()
            elif choice == '3':
                self.remove_model()
            elif choice == '4':
                self.run_simulation()
            elif choice == '5':
                self.check_results()
            elif choice == '6':
                break
            else:
                print("Invalid choice, please try again.")

    def createNewModel(Self):
        # Placeholder for creating a new model
        model_name = input("Enter the name of the new model: ")
        self.models.append(model_name)
        print(f"Model '{model_name}' created and added to the list.")
    
    def startSimulation(self):
        # Placeholder for starting a simulation
        if not self.models:
            print("No models available to run a simulation.")
            return
        print("Starting simulation with the following models:")
        for model in self.models:
            print(f"- {model}")
        # Simulate running the models (placeholder)
        print("Simulation started...")