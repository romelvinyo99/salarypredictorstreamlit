import streamlit as st
import main
import datetime

login_credentials = {
    "Usernames": ["FranklinNthenya1", "ElvisAsiki2", "RooneyWanjohi3", "MelvynNyakoeNyasani4", "AllanVikiru"],
    "Passwords": ["franko@123", "elvo@123", "rooney@123", "melo@123", "viki@123"]
}


def checkCreds(username, password):
    if username in login_credentials["Usernames"]:
        index = login_credentials["Usernames"].index(username)
        if password in login_credentials["Passwords"][index]:
            return True
        else:
            st.error("Invalid passwords")
            return False
    else:
        st.error("Invalid username")
        return False


def login():
    if "tries" not in st.session_state:
        st.session_state.tries = 4
    if "login_success" not in st.session_state:
        st.session_state.login_success = False

    if st.session_state.tries > 1:
        username = st.text_input("Username: ")
        password = st.text_input("Password: ", type="password")
        if st.button("login"):
            if checkCreds(username, password):
                st.session_state.login_success = True
                st.success("login successful")
                st.session_state.tries = 4
                return True
            else:
                st.session_state.tries -= 1
                if st.session_state.tries > 0:
                    st.warning(f"{st.session_state.tries} attempts remaining")
                else:
                    st.error("Access Denied")
                    st.stop()
    else:
        st.error("Access Denied")


if "login_success" in st.session_state and st.session_state.login_success:
    main.main()

else:
    st.warning("Please Login to access site")
    login()
