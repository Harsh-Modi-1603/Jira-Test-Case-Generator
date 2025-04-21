import { AxiosInstance } from "axios";
import axios from "axios";

let axiosInstance: AxiosInstance | undefined;

export function getInstance() {
    if (axiosInstance === undefined) {
        axiosInstance = axios.create({
            baseURL: "http://localhost:8000",
        });
    }
    return axiosInstance;
}
