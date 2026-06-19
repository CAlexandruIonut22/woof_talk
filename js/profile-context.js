const PROFILE_KEY = "woof-talk-dog-profile-v1";

function value(id) {
  return document.getElementById(id)?.value?.trim() || "";
}

function checkedValues(selector) {
  return [...document.querySelectorAll(selector)]
    .filter((input) => input.checked)
    .map((input) => input.value);
}

export function getDogProfile() {
  return {
    name: value("dogName"),
    breed: value("dogBreed"),
    age: value("dogAge"),
    baseline: value("dogBaseline")
  };
}

export function getSceneContext() {
  return {
    trigger: value("sceneTrigger"),
    bodyLanguage: checkedValues("[data-body-language]"),
    notes: value("sceneNotes")
  };
}

export function saveDogProfile() {
  const profile = getDogProfile();
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  return profile;
}

export function hydrateDogProfile() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROFILE_KEY) || "{}");

    if (saved.name) document.getElementById("dogName").value = saved.name;
    if (saved.breed) document.getElementById("dogBreed").value = saved.breed;
    if (saved.age) document.getElementById("dogAge").value = saved.age;
    if (saved.baseline) document.getElementById("dogBaseline").value = saved.baseline;
  } catch {
    // Ignore broken localStorage data.
  }
}